# Third-party tabletop assets

This file records the origin and license of third-party assets vendored in this directory.

## Google Scanned Objects MuJoCo models

Source repository: <https://github.com/kevinzakka/mujoco_scanned_objects>

Pinned source commit: `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`

Vendored object directories:

- `Marc_Anthony_True_Professional_Strictly_Curls_Curl_Defining_Lotion`
- `Android_Figure_Orange`
- `Cole_Hardware_Mug_Classic_Blue`
- `3M_Vinyl_Tape_Green_1_x_36_yd`
- `Kotobuki_Saucer_Dragon_Fly`
- `Room_Essentials_Salad_Plate_Turquoise`
- `Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring`
- `Cole_Hardware_Bowl_Scirocco_YellowBlue`
- `Room_Essentials_Bowl_Turquiose`
- `Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl`
- `Nestle_Nips_Hard_Candy_Peanut_Butter`
- `Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box`

The source repository states that its OBJ and PNG 3D assets are licensed under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/), and its MJCF XML files are licensed under the MIT License. Attribution: Google Scanned Objects dataset; MuJoCo conversion by Kevin Zakka. The scene generator preserves the source geometry and textures, namespaces the MJCF names, and supplies explicit approximate household masses.

Original dataset paper:

> Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman,
> Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. “Google Scanned
> Objects: A High-Quality Dataset of 3D Scanned Household Items.” 2022.
> <https://arxiv.org/abs/2204.11918>

The scans may contain product names, packaging, or trademarks. Their inclusion here identifies the scanned source object and does not imply endorsement.

## Kenney Food Kit fruit models

Asset page: <https://kenney.nl/assets/food-kit>

Official archive: <https://kenney.nl/media/pages/assets/food-kit/83086fa91c-1719418518/kenney_food-kit.zip>

Imported models: `apple`, `orange`, `banana`, `lemon`, and `pear` from Food Kit 2.0. The original OBJ geometry is rotated from Y-up to Z-up and normalized to realistic household dimensions by `scripts/import_kenney_fruit.py`. The original palette texture and license are preserved under `objects/kenney_food_kit/`.

License: [Creative Commons Zero 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). Kenney explicitly permits personal, educational, and commercial use, with attribution optional. Attribution is nevertheless included here: **Food Kit 2.0 by Kenney**.

Downloaded from the official asset page on 2026-08-06. Archive SHA-256: `cdad90853682499b94c9fda2f87678b24bfd8f3264e0ed323f6b6a27fd7c6f6f`.

## Optional Sketchfab fruit scans

Higher-detail `apple_scan` and `orange_scan` alternatives can be downloaded through Sketchfab's official authenticated Download API by `scripts/import_sketchfab_fruit.py`. The script converts glTF geometry and texture data to MuJoCo-compatible OBJ, PNG, and MJCF files, normalizes the models to household scale, and preserves an attribution file beside each converted model.

- Apple: [Sketchfab model `0176be079c2449e7aaebfb652910a854`](https://sketchfab.com/models/0176be079c2449e7aaebfb652910a854), creator **-Arthur-**, scan derived from work by Moshe Caine, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Orange: [Sketchfab model `a35bb9551eeb4da799bf2d0c70b5a54c`](https://sketchfab.com/models/a35bb9551eeb4da799bf2d0c70b5a54c), creator **LR-Scans**, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Sketchfab requires each user to authenticate model downloads. The repository does not contain, log, or persist `SKETCHFAB_TOKEN`; these optional scans are therefore not vendored by default.

## Poly Haven wood texture

Asset page: <https://polyhaven.com/a/wood_table>

Downloaded file: `textures/wood_table_diff_1k.jpg`

Official download URL: <https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_table/wood_table_diff_1k.jpg>

MD5: `45af5766b300f7e940ee8984c5fabd4f`

`wood_table_diff_1k.png` is a lossless PNG encoding of the downloaded JPG because MuJoCo expects PNG textures. Poly Haven publishes its assets under [CC0](https://polyhaven.com/license). Attribution is not required, but is included here for provenance.

Downloaded and verified on 2026-08-05.
