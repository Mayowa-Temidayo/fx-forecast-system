from __future__ import annotations

print("1. Script started")

try:
    print("2. Importing settings...")
    from fx_forecast.config.settings import settings

    print("3. Settings imported")

    print("4. Importing downloader...")
    from fx_forecast.data.fetch import download_fx_data

    print("5. Downloader imported")

except Exception:
    import traceback

    traceback.print_exc()
    raise

print("6. Currency pairs:", settings.currency_pairs)


def main() -> None:
    print("7. Inside main()")

    for pair in settings.currency_pairs:
        print(f"Downloading {pair}")
        path = download_fx_data(
            ticker=pair,
            start_date=settings.start_date,
            interval=settings.interval,
        )
        print(path)


if __name__ == "__main__":
    main()
