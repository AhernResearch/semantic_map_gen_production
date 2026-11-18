import run_config as rc
"""GPU open-vocabulary retrieval on fixed-stride sampled NV12 images."""
from pathlib import Path
import json
import numpy as np
import open_clip
import torch
from PIL import Image,ImageDraw,ImageFont

ROOT=rc.WORKSPACE;DATA=ROOT/'supplied_dataset';OUT=ROOT/'semantic_keyframes';OUT.mkdir(exist_ok=True)
rows=json.loads((DATA/'camera_samples.json').read_text());poses=np.load(DATA/'camera_poses.npz')['poses']
labels=['door','electrical cabinet','fire extinguisher','traffic cone','cardboard box','stairs','person','chair','railing','window','corridor','robot','bed']
model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32-quickgelu',pretrained='openai',device='cuda',cache_dir=str(rc.CLIP_CACHE));model.eval();features=[]
with torch.inference_mode():
    text=model.encode_text(open_clip.get_tokenizer('ViT-B-32-quickgelu')([f'a photo of a {x}' for x in labels]).to('cuda')).float();text=torch.nn.functional.normalize(text,dim=-1).cpu().numpy()
    for start in range(0,len(rows),24):
        images=[]
        for row in rows[start:start+24]:
            with Image.open(DATA/'rgb_rectified'/f'{row["file"]}.png') as im:images.append(preprocess(im.convert('RGB')))
        emb=model.encode_image(torch.stack(images).to('cuda')).float();emb=torch.nn.functional.normalize(emb,dim=-1);features.append(emb.cpu().numpy());print('encoded',min(start+24,len(rows)),flush=True)
features=np.concatenate(features);scores=features@text.T
np.savez_compressed(OUT/'clip_features.npz',image_features=features,text_features=text,cosine_scores=scores,labels=np.asarray(labels),poses_world_from_camera=poses)
retrieval={label:[{'frame':int(i),'source_frame':rows[i]['source_index'],'image':f'../supplied_dataset/rgb_rectified/{rows[i]["file"]}.png','cosine':float(scores[i,j]),'camera_viewpoint_xyz':poses[i,:3,3].tolist()} for i in np.argsort(-scores[:,j])[:8]] for j,label in enumerate(labels)}
(OUT/'retrieval.json').write_text(json.dumps(retrieval,indent=2))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);sheet=Image.new('RGB',(1200,710),'#f5f6f8');draw=ImageDraw.Draw(sheet)
for j,label in enumerate(['door','electrical cabinet','fire extinguisher','traffic cone','stairs','person']):
    best=retrieval[label][0];x=j%3*400;y=j//3*355
    with Image.open(DATA/'rgb_rectified'/f'{best["frame"]:06d}.png') as im:im.thumbnail((390,270));sheet.paste(im,(x+5,y+45))
    draw.text((x+10,y+7),f'{label} | {best["cosine"]:.3f}',font=font,fill='#20242a');draw.text((x+10,y+325),f'frame {best["source_frame"]}',font=font,fill='#20242a')
sheet.save(OUT/'retrieval_preview.jpg',quality=94)
print('SEMANTIC_CACHE_READY',features.shape,flush=True)
