import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                               QWidget, QComboBox, QLabel, QMessageBox, QHBoxLayout, QFrame)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from nsepython import  *
from PySide6.QtGui import QPalette, QColor, QFont

nifty_50 = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050')
nifty_largemidcap250=nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20LARGEMIDCAP%20250')
# Print the stock symbols


def calculate_ema(data, window=21):
    ema = data.ewm(span=window, adjust=False).mean()
    return ema


# Function to get Index stocks

def get_index_stocks(stocks):
    stock_list=[]
    for stock in stocks['data']:
        stock_list.append(stock['symbol'])
        stock_list.sort()
    return stock_list

stock_list=get_index_stocks(nifty_largemidcap250)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Use exponential moving average for average gain and loss
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi




# Main Window class
class StockRSIPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock RSI Plotter")
        self.setGeometry(100, 100, 900, 700)

        # Set Fusion style and dark palette
        QApplication.setStyle("Fusion")
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(dark_palette)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Top section with label and dropdown
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        self.ticker_label = QLabel("Select Stock from the Dropdown:")
        self.ticker_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.ticker_label.setStyleSheet("color: #00BFFF;")

        self.ticker_dropdown = QComboBox()
        self.ticker_dropdown.setFont(QFont("Segoe UI", 11))
        self.ticker_dropdown.setStyleSheet(
            "QComboBox { background-color: #232323; color: #fff; border-radius: 5px; padding: 5px; }"
        )
        self.ticker_dropdown.addItem("Select Stock from the List")
        self.ticker_dropdown.setItemData(0, 0)
        self.ticker_dropdown.addItems(stock_list)
        self.ticker_dropdown.currentIndexChanged.connect(self.on_stock_change)

        top_layout.addWidget(self.ticker_label)
        top_layout.addWidget(self.ticker_dropdown)
        main_layout.addLayout(top_layout)

        # Loading label
        self.loading_label = QLabel("Loading...", alignment=Qt.AlignCenter)
        self.loading_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.loading_label.setStyleSheet("color: #FFA500;")
        self.loading_label.setVisible(False)
        main_layout.addWidget(self.loading_label)

        # Plot area frame for modern look
        self.plot_frame = QFrame()
        self.plot_frame.setFrameShape(QFrame.StyledPanel)
        self.plot_frame.setStyleSheet("background-color: #232323; border-radius: 10px;")
        plot_layout = QVBoxLayout(self.plot_frame)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        self.plot_canvas = None
        main_layout.addWidget(self.plot_frame, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.ticker_dropdown.setFocus()

    def on_stock_change(self):
        if self.ticker_dropdown.currentIndex() == 0:
            return
        self.loading_label.setVisible(True)
        QTimer.singleShot(100, self.plot_rsi)

    def plot_rsi(self):
        ticker = self.ticker_dropdown.currentText().strip().upper()
        if not ticker:
            QMessageBox.critical(self, "Input Error", "Please select a stock ticker symbol")
            self.loading_label.setVisible(False)
            return

        try:
            data = yf.download(f"{ticker}.NS", period='15y', interval='1d', auto_adjust=True)
            data = data.tail(5475)
            if data.empty:
                raise ValueError("No data found for the ticker symbol")

            data['RSI'] = calculate_rsi(data['Close'])
            data['EMA_RSI'] = calculate_ema(data['RSI'])

            weekly_data = data['Close'].resample('W').last()
            weekly_data_ema21 = calculate_ema(weekly_data,21)
            weekly_data_ema50 = calculate_ema(weekly_data,50)
            weekly_rsi = calculate_rsi(weekly_data)
            weekly_rsi_ema = calculate_ema(weekly_rsi)

            monthly_data = data['Close'].resample('ME').last()
            monthly_rsi = calculate_rsi(monthly_data)
            monthly_rsi_ema = calculate_ema(monthly_rsi)

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
            fig.tight_layout(pad=3.0)

            # Plotting logic (same as before)...
            ax1.plot(weekly_data.index, weekly_data, label='Weekly Close Price', color='black')
            # ax1.plot(weekly_data_ema21.index, weekly_data_ema21, label='21 EMA', color='green')
            ax1.plot(weekly_data_ema50.index, weekly_data_ema50, label='50 EMA', color='red')
            # ax1.plot(data.index, data['Close'], label='Daily Close Price', color='black')
            ax1.grid()
            ax1.margins(x=0.1, y=0.5)
            ax1.set_title(f'{ticker} Weekly Close Price')
            ax1.set_ylabel('Price')
            ax1.legend()

            ax2.plot(data.index, data['RSI'], label='Daily RSI', color='orange')
            ax2.plot(data.index, data['EMA_RSI'], label='EMA over RSI', color='black')
            ax2.margins(x=0.1, y=0.5)
            ax2.axhline(40, color='red', linestyle='--')
            ax2.axhline(50, color='grey', linestyle='--')
            ax2.axhline(60, color='green', linestyle='--')
            ax2.set_title('Daily RSI')
            ax2.set_ylabel('RSI')
            ax2.legend()

            weekly_line, = ax3.plot(weekly_data.index, weekly_rsi, label='Weekly RSI', color='orange')
            ax3.plot(weekly_data.index, weekly_rsi_ema, label='EMA over RSI', color='black')
            ax3.axhline(40, color='red', linestyle='--')
            ax3.axhline(50, color='grey', linestyle='--')
            ax3.axhline(60, color='green', linestyle='--')
            ax3.margins(x=0.1, y=0.5)
            ax3.set_title('Weekly RSI')
            ax3.set_ylabel('RSI')
            ax3.legend()

            monthly_line, = ax4.plot(monthly_data.index, monthly_rsi, label='Monthly RSI', color='red')
            ax4.plot(monthly_data.index, monthly_rsi_ema, label='EMA over RSI', color='black')
            ax4.margins(x=0.1, y=0.5)
            ax4.axhline(40, color='red', linestyle='--')
            ax4.axhline(50, color='grey', linestyle='--')
            ax4.axhline(60, color='green', linestyle='--')
            ax4.set_title('Monthly RSI')
            ax4.set_ylabel('RSI')
            ax4.legend()

            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xlim(min(data.index), max(data.index))

            # Hover annotations (same as before)...
            weekly_annot = ax3.annotate("", xy=(0, 0), xytext=(20, 20),
                                        textcoords="offset points",
                                        bbox=dict(boxstyle="round", fc="w"),
                                        arrowprops=dict(arrowstyle="->"))
            weekly_annot.set_visible(False)

            monthly_annot = ax4.annotate("", xy=(0, 0), xytext=(20, 20),
                                         textcoords="offset points",
                                         bbox=dict(boxstyle="round", fc="w"),
                                         arrowprops=dict(arrowstyle="->"))
            monthly_annot.set_visible(False)

            # Add annotation for ax1 (Weekly Close Price)
            price_annot = ax1.annotate("", xy=(0, 0), xytext=(20, 20),
                                       textcoords="offset points",
                                       bbox=dict(boxstyle="round", fc="w"),
                                       arrowprops=dict(arrowstyle="->"))
            price_annot.set_visible(False)

            def update_price_annot(ind):
                # if not ind["ind"]:
                #     return
                x = weekly_data.index.to_numpy()
                y = weekly_data.values
                idx = ind["ind"][0]
                price_annot.xy = (x[idx], y[idx])
                date_str = pd.to_datetime(x[idx]).strftime('%d-%m-%Y')
                val = y[idx]
                if isinstance(val, (np.ndarray, pd.Series)):
                    val = float(val)
                text = f"{date_str}\nPrice: {val:.2f}"
                price_annot.set_text(text)
                price_annot.get_bbox_patch().set_facecolor("yellow")
                price_annot.get_bbox_patch().set_alpha(0.8)

            def update_weekly_annot(ind):
                x, y = weekly_line.get_data()
                weekly_annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
                date_str = pd.to_datetime(x[ind['ind'][0]]).strftime('%d-%m-%Y')
                text = f"{date_str}\nRSI: {y[ind['ind'][0]]:.2f}"
                weekly_annot.set_text(text)
                weekly_annot.get_bbox_patch().set_facecolor("yellow")
                weekly_annot.get_bbox_patch().set_alpha(0.8)

            def update_monthly_annot(ind):
                x, y = monthly_line.get_data()
                monthly_annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
                date_str = pd.to_datetime(x[ind['ind'][0]]).strftime('%d-%m-%Y')
                text = f"{date_str}\nRSI: {y[ind['ind'][0]]:.2f}"
                monthly_annot.set_text(text)
                monthly_annot.get_bbox_patch().set_facecolor("yellow")
                monthly_annot.get_bbox_patch().set_alpha(0.8)

            def hover(event):
                vis_price = price_annot.get_visible()
                vis_weekly = weekly_annot.get_visible()
                vis_monthly = monthly_annot.get_visible()
                if event.inaxes == ax1:
                    cont, ind = ax1.lines[0].contains(event)
                    if cont:
                        update_price_annot(ind)
                        price_annot.set_visible(True)
                        fig.canvas.draw_idle()
                    elif vis_price:
                        price_annot.set_visible(False)
                        fig.canvas.draw_idle()
                elif event.inaxes == ax3:
                    cont, ind = weekly_line.contains(event)
                    if cont:
                        update_weekly_annot(ind)
                        weekly_annot.set_visible(True)
                        fig.canvas.draw_idle()
                    elif vis_weekly:
                        weekly_annot.set_visible(False)
                        fig.canvas.draw_idle()
                elif event.inaxes == ax4:
                    cont, ind = monthly_line.contains(event)
                    if cont:
                        update_monthly_annot(ind)
                        monthly_annot.set_visible(True)
                        fig.canvas.draw_idle()
                    elif vis_monthly:
                        monthly_annot.set_visible(False)
                        fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

            # Remove old canvas if exists
            plot_layout = self.plot_frame.layout()
            if self.plot_canvas:
                plot_layout.removeWidget(self.plot_canvas)
                self.plot_canvas.deleteLater()
                self.plot_canvas = None

            self.plot_canvas = FigureCanvas(fig)
            plot_layout.addWidget(self.plot_canvas)
            self.plot_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error for {ticker}: {str(e)}")
        finally:
            self.loading_label.setVisible(False)


# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set modern font globally
    app.setFont(QFont("Segoe UI", 10))
    main_window = StockRSIPlotter()
    main_window.show()
    sys.exit(app.exec())
