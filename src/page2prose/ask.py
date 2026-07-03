import argparse
import sys

from dotenv import load_dotenv

from .agent.loop import answer

load_dotenv()


def _print_step(event: dict) -> None:
    if event["type"] == "tool_call":
        print(f"  → {event['name']}({event.get('args', {})})", file=sys.stderr)
    elif event["type"] == "tool_result":
        print(f"    {event['name']}: {event['summary']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question over the extracted guidelines.")
    parser.add_argument("query")
    args = parser.parse_args(argv)

    result = answer(args.query, on_event=_print_step)
    if not result.complete or not result.answer:
        print("No answer (the agent could not complete or found nothing relevant).")
        return 0

    print(result.answer)
    if result.citations:
        print("\nSources:")
        for c in result.citations:
            title = f" — {c['title']}" if c.get("title") else ""
            print(f"  {c['guideline_id']} p.{c['page_number']}{title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
