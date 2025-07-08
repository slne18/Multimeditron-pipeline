
import openai
import time

openai.api_key = "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"  # ou config.OPENAI_API_KEY



batch_ids = [
    "batch_685af89074f8819080fc784c1075f49c",
    "batch_685af8eb919c81909324e03ef4885657",
    "batch_685af90d8c38819099336f6fdec2ab7b",
    "batch_685af93b37048190b95d941f69b9b086"
]

import json



while True:
    for batch_id in batch_ids:
        batch = openai.batches.retrieve(batch_id)
        status = batch.status
        print(f"🧾 Batch: {batch_id}")
        print(f"   ↳ Status: {status}")

        if hasattr(batch, "request_counts"):
            counts = batch.request_counts
            print(f"   ↳ Total:     {counts.total}")
            print(f"   ↳ Completed: {counts.completed}")
            print(f"   ↳ Failed:    {counts.failed}")
        else:
            print("   ↳ No request counts available.")

        if status == "failed":
            print("❌ Entire batch failed.")
        elif status == "completed":
            if batch.output_file_id:
                print(f"✅ Output File ID: {batch.output_file_id}")
            else:
                print("⚠️ Completed but no output file was generated (all requests may have failed).")


    print("---")
    time.sleep(60)
