import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import json



def load_image(path, base_dir, image_size=256):
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)

    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.uint8)


def build_h5_from_manifest(full_seq_manifest, base_dir, output_path, image_size=256):
    sequences = []

    for wound_id, day_dict in tqdm(full_seq_manifest.items(), desc="Processing wounds"):
        sorted_days = sorted(day_dict.keys(), key=lambda d: int(d.split('_')[1]))

        image_lists = [day_dict[day] for day in sorted_days]
        combined = list(zip(*image_lists))  # burst-aligned sequences

        if len(combined) == 0:
            continue

        for burst_idx in range(len(combined)):
            seq_paths = list(combined[burst_idx])  # length T
            sequences.append((wound_id, burst_idx, sorted_days, seq_paths))

    print(f"Total sequences: {len(sequences)}")

    # --- Write HDF5 ---
    with h5py.File(output_path, "w") as f:
        len_group = f.create_group("len")

        for seq_idx, (wound_id, burst_idx, sorted_days, seq_paths) in enumerate(tqdm(sequences, desc="Writing HDF5")):
            seq_group = f.create_group(str(seq_idx))

            # --- metadata ---
            seq_group.attrs["wound_id"] = wound_id
            seq_group.attrs["burst_idx"] = burst_idx
            seq_group.attrs["days"] = np.array(sorted_days, dtype="S")

            image_names = [os.path.basename(p) for p in seq_paths]
            seq_group.attrs["image_names"] = np.array(image_names, dtype="S")

            T = len(seq_paths)
            len_group.create_dataset(str(seq_idx), data=T)

            for frame_idx, img_path in enumerate(seq_paths):
                img = load_image(img_path, base_dir, image_size)

                seq_group.create_dataset(
                    str(frame_idx),
                    data=img,
                    dtype="uint8",
                    compression="gzip"
                )

    print(f"Saved to {output_path}")

def pig_id(wound_id):
    # "ID1326_Wound_I" -> "ID1326"
    return wound_id.split("_")[0]

def create_datasets_FULL(args, val_id = None):
    """CREATE THE FULL DATASETS; LEN(SEQ) >4; CREATED 2/5/26"""
    with open(args.man_pth, "r") as f:
        manifest_dict = json.load(f)



    exclude_pigs = {"ID1325", "ID1328"}

    if val_id is not None:
        exclude_pigs.add(val_id)

    all_sequences = list(manifest_dict.keys())

    # Keep only sequences whose pig is NOT in excluded set
    partial_full_seq_ids = [
        w for w in all_sequences
        if pig_id(w) not in exclude_pigs
    ]

    partial_full_manifest = {
        k: v for k, v in manifest_dict.items()
        if (k in partial_full_seq_ids and len(v) > 4)
    }

    val_manifest = {
        k: v for k, v in manifest_dict.items()
        if (pig_id(k) in exclude_pigs and len(v) > 4)   # or however you want to define val now
    }


    for wound_id in partial_full_manifest:
        for day in partial_full_manifest[wound_id]:
            partial_full_manifest[wound_id][day] = [
                clip_after_512(p) for p in partial_full_manifest[wound_id][day]
            ]

    for wound_id in val_manifest:
        for day in val_manifest[wound_id]:
            val_manifest[wound_id][day] = [
                clip_after_512(p) for p in val_manifest[wound_id][day]
            ]

    return partial_full_manifest, val_manifest

def clip_after_512(path):
    parts = path.split("512x512/")
    return parts[1] if len(parts) > 1 else path

class Args:
    man_pth = '/content/drive/MyDrive/Heals_Winter_24/JMIR Heals/manifest.json'
    base_dir = '/content/512x512'
    output_path = "/content/data/val/davinci_val.h5"


args = Args()


partial_full_manifest, val_manifest = create_datasets_FULL(args)

# !mkdir /content/data
# !mkdir /content/data/train
# !mkdir /content/data/val

build_h5_from_manifest(full_seq_manifest=val_manifest,
                       base_dir=args.base_dir,
                       output_path=args.output_path,
                       image_size=256)