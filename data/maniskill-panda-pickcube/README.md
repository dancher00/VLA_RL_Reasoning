---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- ManiSkill
- Panda
- pick-and-place
- manipulation
configs:
- config_name: default
  data_files: data/*/*.parquet
---

## Data Format

The dataset follows the LeRobot standard format.

## Dataset Structure

```
dancher00/maniskill-panda-pickcube/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.main/
│           ├── episode_000000.mp4
│           ├── episode_000001.mp4
│           └── ...
├── meta/
│   ├── episodes.jsonl
│   ├── info.json
│   ├── stats.json
│   └── tasks.jsonl
└── README.md
```


### Data Collection
- **Simulator**: ManiSkill 3.0
- **Environment**: PickCube-v1

## License

This dataset is released under the Apache 2.0 license.