# Data

`aa6082_bbd_surface_roughness.csv` contains the 15 runs of the three-factor Box–Behnken design.

| Column | Meaning | Unit |
|---|---|---|
| `run` | Run number in standard order | – |
| `N` | Spindle speed | rpm |
| `f` | Feed rate | mm/min |
| `d` | Axial depth of cut | mm |
| `Ra` | Arithmetic mean roughness | µm |

**Source.** The machining experiments and roughness measurements were carried out in 2016 by Maradana Siva
for his M.Tech dissertation at Gayatri Vidya Parishad College of Engineering (Autonomous), Visakhapatnam,
India. The present repository re-analyses those measurements.

**Correction.** Run 7 is analysed at N = 1500 rpm, the level set by the Box–Behnken design. The reasons are
explained in Section 2.2 of the paper.

**Setup (summary).** Material: AA6082-T651 plate, 12 mm thick. Tool: 10 mm solid carbide end mill, slot milling,
one tool used for all runs, randomised run order, dry cutting. Roughness tester and cut-off: see Section 2.3 of the paper.

The data are released under CC BY 4.0. Please cite the paper and acknowledge the original dissertation when
reusing them.
