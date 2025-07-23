import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                               QWidget, QLabel, QMessageBox, QListWidget, QHBoxLayout)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# List of NIFTY50 stocks
nifty50_stocks = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK",
    "KOTAKBANK", "ITC", "LT", "AXISBANK", "HDFC", "SBIN", "BHARTIARTL",
    "ASIANPAINT", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TITAN",
    "ULTRACEMCO", "NESTLEIND", "WIPRO", "ONGC", "POWERGRID",
    "COALINDIA", "M&M", "GRASIM", "HINDALCO", "NTPC", "TATASTEEL",
    "BAJAJFINSV", "ADANIPORTS", "BPCL", "BRITANNIA", "CIPLA",
    "DIVISLAB", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "INDUSINDBK",
    "JSWSTEEL", "SHREECEM", "TATAMOTORS", "TECHM", "UPL", "APOLLOHOSP",
    "BAJAJ-AUTO", "SBILIFE", "HDFCLIFE", "ADANIENT"
]


# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Main Window class
class StockRSIPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock RSI Plotter")
        self.setGeometry(100, 100, 800, 600)

        # Create main layout
        main_layout = QVBoxLayout()

        # Create a horizontal layout for label and list
        top_layout = QHBoxLayout()

        # Create and add label and list widgets
        self.ticker_label = QLabel("Select Stock from NIFTY50:")
        self.ticker_list = QListWidget()
        self.ticker_list.addItems(nifty50_stocks)
        self.ticker_list.currentItemChanged.connect(self.plot_rsi)  # Auto-submit on selection

        # Add label and list to the horizontal layout
        top_layout.addWidget(self.ticker_label)
        top_layout.addWidget(self.ticker_list)

        # Add the horizontal layout to the main layout
        main_layout.addLayout(top_layout)

        # Set the layout in a QWidget
        container = QWidget()
        container.setLayout(main_layout)

        # Set the central widget
        self.setCentralWidget(container)

    def plot_rsi(self):
        selected_item = self.ticker_list.currentItem()
        if not selected_item:
            return

        ticker = selected_item.text().strip().upper()
        if not ticker:
            QMessageBox.critical(self, "Input Error", "Please select a stock ticker symbol")
            return

        try:
            data = yf.download(f"{ticker}.NS", start='2023-01-01', end='2023-12-31')
            if data.empty:
                QMessageBox.critical(self, "Data Error", "No data found for the ticker symbol")
                return

            # Calculate RSI
            data['RSI_Daily'] = calculate_rsi(data['Close'])

            weekly_data = data['Close'].resample('W').last()
            weekly_rsi = calculate_rsi(weekly_data)

            monthly_data = data['Close'].resample('M').last()
            monthly_rsi = calculate_rsi(monthly_data)

            # Create plots
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

            # Daily Close Price Plot
            ax1.plot(data.index, data['Close'], label='Daily Close Price', color='black')
            ax1.set_title('Daily Close Price')
            ax1.set_ylabel('Price')
            ax1.legend()

            # Daily RSI Plot
            ax2.plot(data.index, data['RSI_Daily'], label='Daily RSI', color='blue')
            ax2.axhline(30, color='red', linestyle='--')
            ax2.axhline(70, color='green', linestyle='--')
            ax2.set_title('Daily RSI')
            ax2.set_ylabel('RSI')
            ax2.legend()

            # Weekly RSI Plot
            ax3.plot(weekly_rsi.index, weekly_rsi, label='Weekly RSI', color='orange')
            ax3.axhline(30, color='red', linestyle='--')
            ax3.axhline(70, color='green', linestyle='--')
            ax3.set_title('Weekly RSI')
            ax3.set_ylabel('RSI')
            ax3.legend()

            # Monthly RSI Plot
            ax4.plot(monthly_rsi.index, monthly_rsi, label='Monthly RSI', color='purple')
            ax4.axhline(30, color='red', linestyle='--')
            ax4.axhline(70, color='green', linestyle='--')
            ax4.setTitle('Monthly RSI')
            ax4.set_ylabel('RSI')
            ax4.legend()

            plt.tight_layout()

            # Display the plot in the PySide6 window
            canvas = FigureCanvas(fig)
            layout = self.centralWidget().layout()
            if layout.count() > 1:  # Remove the old plot if it exists
                old_canvas = layout.itemAt(1).widget()
                layout.removeWidget(old_canvas)
                old_canvas.deleteLater()
            layout.addWidget(canvas)
            canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = StockRSIPlotter()
    main_window.show()
    sys.exit(app.exec())
