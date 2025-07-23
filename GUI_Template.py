import sys
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                               QWidget, QComboBox, QLabel, QMessageBox, QHBoxLayout)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from nsepython import  *

# nifty_500 = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500')
nifty_50 = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050')
# nifty_midcap_50 = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500')
# nifty_auto = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20AUTO')
# nifty_bank = nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK')
nifty_largemidcap250=nsefetch('https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20LARGEMIDCAP%20250')
# Print the stock symbols


# List of NIFTY50 stocks
# nifty50_stocks = [
#     "CIPLA","BAJFINANCE","M&M","DIVISLAB","NTPC","HEROMOTOCO","BAJAJFINSV",
#     "POWERGRID","SUNPHARMA","BHARTIARTL","TCS","APOLLOHOSP","TITAN","ULTRACEMCO",
#     "INFY","BAJAJ-AUTO","DRREDDY","ICICIBANK","EICHERMOT","GRASIM","INDUSINDBK",
#     "HDFCLIFE","LT","SBILIFE","BRITANNIA","ASIANPAINT","JSWSTEEL","ADANIENT","ADANIPORTS",
#     "TATACONSUM","LTIM","AXISBANK","SHRIRAMFIN","TATASTEEL","SBIN","ONGC","KOTAKBANK","BPCL",
#     "WIPRO","MARUTI","NESTLEIND","HINDALCO","HINDUNILVR","HCLTECH","ITC","RELIANCE","COALINDIA",
#     "TECHM","HDFCBANK","TATAMOTORS","MARICO"
# ]


print(nifty_50)

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
# print(stock_list)
# stock_list=['BPCL','CPSEETF','HAL','HCLTECH','HDFCBANK','ICICIBANK','IRCTC'
# ,'JIOFIN','MAFANG','RAILTEL','TATACHEM','TATAMOTORS']
# stock_list.sort()

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
        self.setGeometry(100, 100, 800, 600)

        # Create main layout
        main_layout = QVBoxLayout()

        # Create a horizontal layout for label and dropdown
        top_layout = QHBoxLayout()

        # Create and add label and dropdown widgets
        self.ticker_label = QLabel("Select Stock from NIFTY50:")

        self.ticker_dropdown = QComboBox()
        self.ticker_dropdown.addItem("Select Stock from the List")
        self.ticker_dropdown.setItemData(0, 0)  # 0 disables the option

        self.ticker_dropdown.addItems(stock_list)
        self.ticker_dropdown.currentIndexChanged.connect(self.plot_rsi)  # Auto-submit on selection
        self.ticker_dropdown.currentIndexChanged.connect(self.start_loading)  # Auto-submit on selection

        # Add label and dropdown to the horizontal layout
        top_layout.addWidget(self.ticker_label)
        top_layout.addWidget(self.ticker_dropdown)

        # Add the horizontal layout to the main layout
        main_layout.addLayout(top_layout)

        # Loading label
        self.loading_label = QLabel("Loading...", alignment=Qt.AlignCenter)
        self.loading_label.setVisible(False)
        main_layout.addWidget(self.loading_label)

        # Set the layout in a QWidget
        container = QWidget()
        container.setLayout(main_layout)

        # Set the central widget
        self.setCentralWidget(container)
        self.ticker_dropdown.setFocus()

    def start_loading(self):
        self.loading_label.setVisible(True)
        QTimer.singleShot(100, self.plot_rsi)  # Small delay to show the loading label

    def plot_rsi(self):
        ticker = self.ticker_dropdown.currentText().strip().upper()
        # print(ticker)
        if not ticker:
            QMessageBox.critical(self, "Input Error", "Please select a stock ticker symbol")
            return

        try:
            data = yf.download(f"{ticker}.NS", period='max',interval='1d',auto_adjust=True)
            data = data.tail(3650)
            # print(data)
            if data.empty:
                QMessageBox.critical(self, "Data Error", "No data found for the ticker symbol")
                return

            # Calculate RSI
            data['RSI'] = calculate_rsi(data['Close'])
            data['EMA_RSI'] = calculate_ema(data['RSI'])


            weekly_data = data['Close'].resample('W').last()
            weekly_rsi = calculate_rsi(weekly_data)
            weekly_rsi_ema = calculate_ema(weekly_rsi)
            # stock = yf.Ticker(f"{ticker}.NS")
            # weekly_data = stock.history(period='10y', interval='1wk')
            # # weekly_data = yf.download(f"{ticker}.NS", period='10y',interval='1wk')
            #
            # #     (data.resample('W').agg({
            # #     'Open': 'first',
            # #     'High': 'max',
            # #     'Low': 'min',
            # #     'Close': 'last',
            # #     'Volume': 'sum'
            # # }))
            # weekly_data['RSI']=calculate_rsi(weekly_data['Close'],14)
            # weekly_data['EMA_RSI']=calculate_ema(weekly_data['RSI'])

            monthly_data = data['Close'].resample('ME').last()
            monthly_rsi = calculate_rsi(monthly_data)
            monthly_rsi_ema = calculate_ema(monthly_rsi)

            # monthly_data = data.resample('M').agg({
            #     'Open': 'first',
            #     'High': 'max',
            #     'Low': 'min',
            #     'Close': 'last',
            #     'Volume': 'sum'
            # })
            # monthly_data['RSI'] = calculate_rsi(monthly_data['Close'])
            # monthly_data['EMA_RSI'] = calculate_ema(monthly_data['RSI'])
            # print(weekly_data[['Close','RSI','EMA_RSI']].tail(15))

            # Create plots
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

            # Daily Close Price Plot
            ax1.plot(data.index, data['Close'], label='Daily Close Price', color='black')
            ax1.grid()
            ax1.margins(x=0.1, y=0.5)
            ax1.set_title(f'{ticker} Daily Close Price')
            ax1.set_ylabel('Price')
            ax1.legend()

            # Daily RSI Plot
            ax2.plot(data.index, data['RSI'], label='Daily RSI', color='orange')
            ax2.plot(data.index, data['EMA_RSI'], label='EMA over RSI', color='black')
            ax2.margins(x=0.1, y=0.5)
            ax2.axhline(40, color='red', linestyle='--')
            ax2.axhline(60, color='green', linestyle='--')
            ax2.set_title('Daily RSI')
            ax2.set_ylabel('RSI')
            ax2.legend()

            # Weekly RSI Plot
            weekly_line, = ax3.plot(weekly_data.index, weekly_rsi, label='Weekly RSI', color='orange')
            ax3.plot(weekly_data.index, weekly_rsi_ema, label='EMA over RSI', color='black')
            ax3.axhline(40, color='red', linestyle='--')
            ax3.axhline(60, color='green', linestyle='--')
            ax3.margins(x=0.1, y=0.5)
            ax3.set_title('Weekly RSI')
            ax3.set_ylabel('RSI')
            ax3.legend()

            # Monthly RSI Plot
            monthly_line, = ax4.plot(monthly_data.index, monthly_rsi, label='Monthly RSI', color='purple')
            # ax4.plot(monthly_rsi.index, monthly_rsi, label='Monthly RSI', color='orange')
            ax4.plot(monthly_data.index, monthly_rsi_ema, label='EMA over RSI', color='black')
            ax4.margins(x=0.1, y=0.5)
            ax4.axhline(40, color='red', linestyle='--')
            ax4.axhline(60, color='green', linestyle='--')
            ax4.set_title('Monthly RSI')
            ax4.set_ylabel('RSI')
            ax4.legend()

            ax1.set_xlim(min(data.index), max(data.index))
            ax2.set_xlim(min(data.index), max(data.index))
            ax3.set_xlim(min(data.index), max(data.index))
            ax4.set_xlim(min(data.index), max(data.index))



            # Hover functionality for Weekly RSI
            weekly_annot = ax3.annotate("", xy=(0, 0), xytext=(20, 20),
                                        textcoords="offset points",
                                        bbox=dict(boxstyle="round", fc="w"),
                                        arrowprops=dict(arrowstyle="->"))
            weekly_annot.set_visible(False)



            def update_weekly_annot(ind):
                x, y = weekly_line.get_data()
                weekly_annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
                date_str = pd.to_datetime(x[ind['ind'][0]]).strftime('%d-%m-%Y')
                text = f"{date_str}\nRSI: {y[ind['ind'][0]]:.2f}"
                weekly_annot.set_text(text)
                weekly_annot.get_bbox_patch().set_facecolor("yellow")
                weekly_annot.get_bbox_patch().set_alpha(0.8)

            plt.tight_layout()
            # Hover functionality for Monthly RSI
            monthly_annot = ax4.annotate("", xy=(0, 0), xytext=(20, 20),
                                 textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w"),
                                 arrowprops=dict(arrowstyle="->"))
            monthly_annot.set_visible(False)

            def update_monthly_annot(ind):
                x, y = monthly_line.get_data()
                monthly_annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
                date_str = pd.to_datetime(x[ind['ind'][0]]).strftime('%d-%m-%Y')
                text = f"{date_str}\nRSI: {y[ind['ind'][0]]:.2f}"
                monthly_annot.set_text(text)
                monthly_annot.get_bbox_patch().set_facecolor("yellow")
                monthly_annot.get_bbox_patch().set_alpha(0.8)

            def hover(event):
                vis_weekly = weekly_annot.get_visible()
                vis_monthly = monthly_annot.get_visible()

                if event.inaxes == ax3:
                    cont, ind = weekly_line.contains(event)
                    if cont:
                        update_weekly_annot(ind)
                        weekly_annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis_weekly:
                            weekly_annot.set_visible(False)
                            fig.canvas.draw_idle()

                elif event.inaxes == ax4:
                    cont, ind = monthly_line.contains(event)
                    if cont:
                        update_monthly_annot(ind)
                        monthly_annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis_monthly:
                            monthly_annot.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

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
