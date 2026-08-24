from wound_forecasting.manifests import clip_after_512, pig_id, select_pigs


def test_identifiers_and_path_normalization():
    assert pig_id("ID1325_Wound_I") == "ID1325"
    assert clip_after_512("/content/512x512/Day_0/image.JPG") == (
        "Day_0/image.JPG"
    )


def test_select_pigs_prevents_split_contamination():
    manifest = {
        "ID1325_Wound_I": {f"Day_{day}": [f"512x512/a/{day}.JPG"] for day in range(5)},
        "ID1323_Wound_I": {f"Day_{day}": [f"512x512/b/{day}.JPG"] for day in range(5)},
        "ID1328_Wound_C": {f"Day_{day}": [f"512x512/c/{day}.JPG"] for day in range(4)},
    }

    selected = select_pigs(manifest, ["ID1325", "ID1328"])

    assert list(selected) == ["ID1325_Wound_I"]
    assert selected["ID1325_Wound_I"]["Day_0"] == ["a/0.JPG"]

