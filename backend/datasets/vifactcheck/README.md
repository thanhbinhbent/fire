# ViFactCheck Dataset

Vietnamese Fact-Checking dataset for evaluating fact-checking systems.

## Source

- **Original Dataset**: [tranthaihoa/vifactcheck](https://huggingface.co/datasets/tranthaihoa/vifactcheck)
- **Paper**: ViFactCheck: Vietnamese Fact-Checking Dataset (under review)

## Statistics

- **Total samples**: 20 (sample data)
- **Language**: Vietnamese
- **Labels**: true/false (binary classification)
- **Domain**: General knowledge, Vietnamese history, geography, culture

## Format

Each line is a JSON object with:
```json
{
  "claim": "Việt Nam là quốc gia có dân số đông nhất Đông Nam Á.",
  "label": "true"
}
```

## Usage

### Programmatic Usage

This dataset can be used with the FIRE evaluation system:

```python
from common.vifactcheck_loader import ViFactCheckLoader

loader = ViFactCheckLoader()
dataset = loader.load_dataset(split="test")
samples = loader.convert_to_fire_format()
```

### Command-Line Usage

Download and convert the complete dataset using the CLI:

```bash
# Download and convert test split
python common/vifactcheck_loader.py --split test --output-dir datasets/vifactcheck

# Download all splits
python common/vifactcheck_loader.py --split all

# Include NEI (Not Enough Info) samples
python common/vifactcheck_loader.py --split test --include-nei

# Show statistics and sample claims
python common/vifactcheck_loader.py --split test --stats --samples 5
```

## Citation

If you use this dataset, please cite:

```
@misc{tranthaihoa2024vifactcheck,
  title={ViFactCheck: A Vietnamese Fact-Checking Dataset},
  author={Tran Thai Hoa},
  year={2024},
  publisher={HuggingFace}
}
```