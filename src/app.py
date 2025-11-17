from ConsoleApp import ConsoleApp
from DashboardApp import DashboardApp


def run_cli():
    app = ConsoleApp()
    app.run()


def run_dashboard():
    dash = DashboardApp()
    dash.run()

def main(): 
    run_cli()

if __name__ == "__main__":
    main()