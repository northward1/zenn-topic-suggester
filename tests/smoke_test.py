import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

try:
    import zenn_topic_suggester
except ImportError:
    zenn_topic_suggester = None


def test_import():
    assert zenn_topic_suggester is not None


def test_load_data():
    df = zenn_topic_suggester.search_topics()
    assert df is not None
    assert df.index.name == "topics"
    assert "記事数" in df.columns


def test_load_data_setting_start_end_lb():
    df = zenn_topic_suggester.search_topics(
        start=datetime.datetime(year=2026, month=3, day=1).astimezone(
            ZoneInfo("Asia/Tokyo")
        ),
        end=datetime.datetime(year=2026, month=3, day=15).astimezone(
            ZoneInfo("Asia/Tokyo")
        ),
        lower_bound=10000,
    )

    assert len(df) == 0
    assert df.index.name == "topics"
    assert "記事数" in df.columns


def test_fetch_database():
    db_path = zenn_topic_suggester.fetch_database()
    assert isinstance(db_path, Path)
    assert db_path.is_file()
