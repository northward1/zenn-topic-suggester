import optuna
import subprocess

answers = {
    "samples/a.md": ["生成ai", "個人開発", "ai駆動開発", "ai", "aiエージェント"],
    "samples/b.md": ["データ分析", "生成ai", "データサイエンス", "ポエム"],
    "samples/c.md": ["markdown", "zenn", "初心者", "初心者向け"],
    "samples/d.md": ["python", "rust", "go", "プログラミング"],
    "samples/e.md": ["キャリア", "チーム開発", "生成ai", "aiエージェント"],
    "samples/f.md": ["oss", "生成ai", "github", "aiエージェント"],
    "samples/g.md": ["claude", "claudecode", "gemini", "openai", "chatgpt"],
    "samples/h.md": ["react", "typescript", "frontend", "nextjs"],
    "samples/i.md": ["database", "postgresql", "mysql", "sqlite"],
    "samples/j.md": [
        "ソフトウェア開発",
        "システム開発",
        "設計",
        "プロジェクトマネジメント",
        "アジャイル",
    ],
}


def objective(trial):
    score = 0

    wc = trial.suggest_float("wc", 0, 10)
    wf = trial.suggest_float("wf", 0, 10)

    for key in answers.keys():
        result = subprocess.run(
            [
                "uv",
                "run",
                "zenn-topic-suggest",
                key,
                "-s",
                "-wc",
                str(wc),
                "-wf",
                str(wf),
            ],
            capture_output=True,
            text=True,
        )

        stdout = result.stdout
        lines = [
            line.lstrip("-").strip()
            for line in stdout.strip().splitlines()
            if line.strip()
        ]
        s = set(lines)

        for val in answers[key]:
            if val in s:
                score += 1

    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print(study.best_params)
