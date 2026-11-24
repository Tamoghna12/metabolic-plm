import pandas as pd


def test_essentiality_threshold_logic() -> None:
    df = pd.DataFrame(
        {
            "growth_ratio": [0.0, 0.03, 0.50, 1.0],
            "status": ["infeasible", "optimal", "optimal", "optimal"],
        }
    )
    is_essential = ((df["status"] != "optimal") | (df["growth_ratio"] < 0.05)).astype(int)
    assert is_essential.tolist() == [1, 1, 0, 0]
