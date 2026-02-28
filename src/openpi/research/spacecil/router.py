"""Language-visual router for task adapter selection.

Routes to task adapters without oracle task ID at inference time.
Input: language embedding + visual summary.
Output: routing weights over registered task adapters.
"""
