from data.repository import HistoryRepository


def test_repository_returns_cached_result(tmp_path):
    db_path = tmp_path / "history.db"
    repo = HistoryRepository(str(db_path))
    expected = {
        "is_bird": True,
        "predictions": [{"species": "x", "common_name": "y", "confidence": 0.9, "features": []}],
        "reasoning": "ok",
    }
    repo.add_record(image_hash="abc123", file_name="file.jpg", result=expected)

    cached = repo.get_latest_success_by_hash("abc123")
    assert cached is not None
    assert cached["predictions"][0]["common_name"] == "y"
