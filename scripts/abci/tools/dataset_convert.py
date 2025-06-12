#!/usr/bin/env python3
"""
Convert conversation JSONL format from one schema to another.

Input format:
{
  "conversation_id": "1934a7fa1ac34dac96c3568f7707b745",
  "conversation": [
    {
      "content": "おはようございます ",
      "role": "user"
    },
    {
      "role": "assistant",
      "content": "おはようございます！今日も一日頑張りましょう！何かお手伝いできることはありますか？ 😊\n"
    }
  ]
}

Output format:
{
  "system": "detailed thinking off",
  "conversations": [
    {
      "from": "user",
      "value": "おはようございます "
    },
    {
      "from": "assistant",
      "value": "おはようございます！今日も一日頑張りましょう！何かお手伝いできることはありますか？ 😊\n"
    }
  ],
  "mask": "user",
}
"""

import json
import argparse
from typing import Dict, List, Any


def convert_conversation(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single conversation from input format to output format.

    Args:
        input_data: Dictionary containing conversation data in input format

    Returns:
        Dictionary in the target output format
    """
    # Extract conversation list
    conversation_list = input_data.get("conversation", [])

    # Convert each turn to the new format
    converted_conversations = []
    for turn in conversation_list:
        converted_turn = {"from": turn["role"], "value": turn["content"]}
        converted_conversations.append(converted_turn)

    # Create the output structure
    output_data = {
        "system": "detailed thinking off",
        "conversations": converted_conversations,
        "mask": "user",
    }

    return output_data


def process_jsonl_file(input_file: str, output_file: str) -> None:
    """
    Process JSONL file and convert each line from input format to output format.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    processed_count = 0
    error_count = 0

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                # Parse input JSON
                input_data = json.loads(line)

                # Convert to output format
                output_data = convert_conversation(input_data)

                # Write to output file
                json.dump(output_data, outfile, ensure_ascii=False)
                outfile.write("\n")

                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1

    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count} conversations")
    if error_count > 0:
        print(f"Errors encountered: {error_count} lines")


def main():
    parser = argparse.ArgumentParser(
        description="Convert conversation JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-jsonl", required=True, help="Path to output JSONL file")

    args = parser.parse_args()

    try:
        process_jsonl_file(args.input_jsonl, args.output_jsonl)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_jsonl}' not found")
    except PermissionError:
        print(f"Error: Permission denied when accessing files")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
