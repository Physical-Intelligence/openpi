"""Sequential task training loop and backward transfer evaluation.

Orchestrates the full continual learning pipeline:
- Task sequencing
- Adapter training per task
- Optional behavior distillation
- Backward transfer evaluation after each task
"""
