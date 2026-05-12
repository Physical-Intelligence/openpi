from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
