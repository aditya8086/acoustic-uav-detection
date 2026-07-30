
# UaVirBASE Dataset Inspector
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import soundfile as sf

DATASET_ROOT = Path(r"C:\Users\CARE\Downloads\Acoustic based drone detection\Audio files\Datasets\UaVirBASE")

def safe_float(v):
    try: return float(str(v).split()[0])
    except: return np.nan

def bit_depth(s):
    return {"PCM_16":16,"PCM_24":24,"PCM_32":32,"FLOAT":32,"DOUBLE":64}.get(s.upper(),"Unknown")

def main():
    folders=sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()])
    print("="*80); print("UaVirBASE DATASET INSPECTOR"); print("="*80)
    print("Folders:",len(folders))
    sr_c=Counter(); ch_c=Counter(); sub_c=Counter(); bd_c=Counter()
    dist_c=Counter(); h_c=Counter(); rot_c=Counter(); mov_c=Counter(); type_c=Counter(); mic_c=Counter()
    durs=[]; temps=[]; hum=[]; pres=[]; wind=[]; rows=[]; missj=[]; missw=[]; corr=[]; total_bytes=0
    for folder in folders:
        j=list(folder.glob("*.json")); w=list(folder.glob("*.wav"))
        if len(j)!=1: missj.append(folder.name); continue
        if len(w)!=1: missw.append(folder.name); continue
        try:
            info=sf.info(str(w[0]))
            meta=json.load(open(j[0],"r",encoding="utf-8"))
        except:
            corr.append(folder.name); continue
        sr_c[info.samplerate]+=1; ch_c[info.channels]+=1; sub_c[info.subtype]+=1; bd_c[bit_depth(info.subtype)]+=1
        dur=info.frames/info.samplerate; durs.append(dur); total_bytes+=w[0].stat().st_size
        d=meta["drone"]; type_c[d["type"]]+=1; dist_c[str(d["distance"])]+=1; h_c[str(d["height"])]+=1; rot_c[str(d["rotation"])]+=1; mov_c[str(d["movement"])]+=1
        wx=meta["weather_data"]["measurements"]
        temps.append(safe_float(wx["air temperature"])); hum.append(safe_float(wx["air humidity"])); pres.append(safe_float(wx["barometric pressure"])); wind.append(safe_float(wx["wind speed"]))
        mic_c[len(meta["microphone_array"]["microphones"])]+=1
        rows.append({"folder":folder.name,"sample_rate":info.samplerate,"channels":info.channels,"bit_depth":bit_depth(info.subtype),"subtype":info.subtype,"duration_sec":round(dur,3),"distance":d["distance"],"height":d["height"]})
    print("Valid:",len(rows)," Missing JSON:",len(missj)," Missing WAV:",len(missw)," Corrupted:",len(corr))
    if not rows: return
    print("Dataset Size (GB):",round(total_bytes/1024**3,3))
    print("Total Hours:",round(sum(durs)/3600,3))
    print("Sample Rates:",dict(sr_c)); print("Channels:",dict(ch_c)); print("Bit Depths:",dict(bd_c)); print("Subtype:",dict(sub_c))
    print("Duration min/mean/max:",round(np.min(durs),3),round(np.mean(durs),3),round(np.max(durs),3))
    print("Distances:",dict(dist_c)); print("Heights:",dict(h_c)); print("Rotations:",dict(rot_c)); print("Movements:",dict(mov_c)); print("Drone Types:",dict(type_c)); print("Microphones:",dict(mic_c))
    pd.DataFrame(rows).to_csv("uavirbase_audio_report.csv",index=False)
    pd.DataFrame({"missing_json":missj,"missing_wav":missw,"corrupted":corr}).to_csv("uavirbase_integrity_report.csv",index=False)
    json.dump({"recordings":len(rows),"sample_rates":dict(sr_c),"channels":dict(ch_c),"bit_depths":dict(bd_c),"distances":dict(dist_c),"heights":dict(h_c)},open("dataset_summary.json","w"),indent=4)
    print("Saved CSVs and dataset_summary.json")
if __name__=="__main__":
    main()
