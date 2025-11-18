from typing import Dict
from cli_app import action_arima_forecast, action_compare_ml, action_corr_heatmap, action_load_info, action_lr_forecast, action_run_eda, action_trend_boxplot, print_menu
from utils import ensure_dir


def main() -> None:
    """Main CLI loop."""
    ensure_dir("data/raw")
    ensure_dir("results")

    state: Dict = {}

    while True:
        print_menu()
        choice = input("Select option: ").strip()

        if choice == "0":
            print("Goodbye.")
            break
        elif choice == "1":
            action_load_info(state)
        elif choice == "2":
            action_run_eda(state)
        elif choice == "3":
            action_trend_boxplot(state)
        elif choice == "4":
            action_corr_heatmap(state)
        elif choice == "5":
            action_arima_forecast(state)
        elif choice == "6":
            action_compare_ml(state)
        elif choice == "7":
            action_lr_forecast(state)
        else:
            print("Invalid choice, try again.")


if __name__ == "__main__":
    main()