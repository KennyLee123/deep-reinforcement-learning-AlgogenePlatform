


from AlgoAPI import AlgoAPIUtil, AlgoAPI_Backtest
from stable_baselines3.ppo import PPO

from collections import deque
import numpy as np
import pandas as pd
import talib

# for fixed decimal calculation
from decimal import *

class InstrumentData:
    '''
    Custom class for storing pass data of cryptos
    '''
    def __init__(self, tic):
        '''
        Initiate InstrumentData for calculate technical indicies
        tic = instrument name
        '''
        self.tic = tic

        # fixed maxlen for deques
        self.maxlen = 20

        # init deque with zero
        self.deque_h = deque([0] * self.maxlen, maxlen=self.maxlen)
        self.deque_l = deque([0] * self.maxlen, maxlen=self.maxlen)
        self.deque_c = deque([0] * self.maxlen, maxlen=self.maxlen)


    def append_data(self, h, l, c):
        '''
        Append latest data of the instrument to the deques
        '''
        self.deque_h.append(h)
        self.deque_l.append(l)
        self.deque_c.append(c)

    def get_rsi(self):
        '''
        Get latest RSI of the instrument
        
        don't know why inconsistencies exist
        '''
        close_array = np.array(list(self.deque_c), dtype = np.dtype(float))
        rsi = talib.RSI(close_array, timeperiod = 14)
        
        return rsi[-1]

    def get_bollinger_bands(self):
        '''
        Get latest bollinger bands of the instrument
        Return in upperBB, middleBB, lowerBB
        
        no inconsistencies exists
        '''
        close_array = np.array(list(self.deque_c), dtype = np.dtype(float))
        upperBB, middleBB, lowerBB = talib.BBANDS(close_array, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        return upperBB[-1], middleBB[-1], lowerBB[-1]

    # KELTNER CHANNEL CALCULATION
    @staticmethod
    def _get_kc(high, low, close, kc_lookback=20, multiplier=2, atr_lookback=10):
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift()))
        tr3 = pd.DataFrame(abs(low - close.shift()))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.ewm(alpha = 1/atr_lookback).mean()
        
        kc_middle = close.ewm(kc_lookback).mean()
        kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
        kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
        
        return kc_middle, kc_upper, kc_lower

    
    def get_keltber_channel(self):
        '''
        Get latest keltber channel of the instrument
        Return in kc_middle, kc_upper, kc_lower
        
        Slight inconsistencies exist
        '''
        close_df = pd.DataFrame(np.array(list(self.deque_c), dtype = np.dtype(float)))
        high_df = pd.DataFrame(np.array(list(self.deque_h), dtype = np.dtype(float)))
        low_df = pd.DataFrame(np.array(list(self.deque_l), dtype = np.dtype(float)))

        kc_middle, kc_upper, kc_lower = InstrumentData._get_kc(high_df.iloc[:, 0], low_df.iloc[:, 0], close_df.iloc[:, 0])

        return kc_middle.iloc[[-1, 0]][0], kc_upper.iloc[[-1, 0]][0], kc_lower.iloc[[-1, 0]][0]

class Trade:
    def __init__(self, tradeID, trade_dict):
        self.tradeID = tradeID
        self.instrument = trade_dict['instrument']
        self.buysell = trade_dict['buysell']
        self.openprice = trade_dict['openprice']
        self.volume = Decimal(trade_dict['Volume']).quantize(Decimal('.01'))


class AlgoEvent:
    def __init__(self):
        pass
        
    def start(self, mEvt):
        self.evt = AlgoAPI_Backtest.AlgoEvtHandler(self, mEvt)
        
        # place our init here
        self.instrument_list = mEvt['subscribeList']
        self.instrument_list.sort()
        self.instrument_name_to_index_dict = {
            inst: i for i, inst in enumerate(self.instrument_list)
        }

        # store the opened orders of each instruments
        # refreshed after every on_bulkdatafeed
        self.instrument_openorders = {
            inst: [] for inst in self.instrument_list
        }

        # create InstrumentData class for each instrument for storing the past open/close/high price
        self.instrument_data_dict = {
            tic: InstrumentData(tic) for tic in self.instrument_list
        }

        ####################
        # model related
        ####################

        # the file name of the model (without .zip)
        self.model_filename = 'model_ohlcbamv_16-10-2022'
        self.model = PPO.load(self.evt.path_lib + self.model_filename)

        # column names of state dataframe of the model
        self.state_df_columns = [
            'date', 'o', 'h', 'l', 'close', 'b', 'a', 'm', 'v', 'tic', 'upperBB', 'middleBB', 'lowerBB', 'rsi', 'kc_middle', 'kc_upper', 'kc_lower'
        ]
        
        
        # _initiate_state related
        # ratio_list = self.state_df_columns - ['date', 'close', 'tic']
        # must be exact ordering as in training environment
        self.ratio_list = ['o','h', 'l', 'b', 'a', 'm', 'v','upperBB','middleBB','lowerBB','rsi','kc_middle','kc_upper','kc_lower']
        self.stock_dimension = len(self.instrument_list)
        self.initial_amount = mEvt['InitialCapital']
        self.initial = True
        self.previous_state = None

        # action related
        # scaling factor of actions from [-1, 1] to [-hmax, hmax]
        self.hmax = 100
        

        # counter for on_bulkdatafeed
        self.counter_flag = 0
    
        # last line of code    
        self.evt.start()
        
    def on_bulkdatafeed(self, isSync, bd, ab):
        self.counter_flag += 1
        # wait until all instruments are updated in there
        if self.counter_flag < self.stock_dimension:
            return
        
        # init
        dict_for_state_df = {
            col: [] for col in self.state_df_columns
        }
        
        for i, instrument in enumerate(sorted(list(bd.keys()))):
            # self.evt.consoleLog(bd[instrument])
            
            high_price = max(0, bd[instrument]["highPrice"])            # max(0, ) for -1 protection
            low_price = max(0, bd[instrument]["lowPrice"])
            close_price = max(0, bd[instrument]["lastPrice"])
            
            
            dict_for_state_df['date'].append(bd[instrument]["timestamp"])
            dict_for_state_df['tic'].append(bd[instrument]['instrument'])
            # open price = last close price if any, or temporary set to current close price
            dict_for_state_df['o'].append(close_price if (self.previous_state == None) else self.previous_state[1 + i])
            dict_for_state_df['close'].append(close_price)        
            dict_for_state_df['h'].append(high_price)
            dict_for_state_df['l'].append(low_price)
            dict_for_state_df['m'].append(max(0, bd[instrument]["midPrice"]))
            dict_for_state_df['v'].append(max(0, bd[instrument]["volume"]))
            dict_for_state_df['b'].append(max(0, bd[instrument]["bidPrice"]))
            dict_for_state_df['a'].append(max(0, bd[instrument]["askPrice"]))

            # update techical indicators
            self.instrument_data_dict[instrument].append_data(high_price, low_price, close_price)

            # get technical indicators
            rsi = self.instrument_data_dict[instrument].get_rsi()
            dict_for_state_df['rsi'].append(rsi)
            kc_middle, kc_upper, kc_lower = self.instrument_data_dict[instrument].get_keltber_channel()
            dict_for_state_df['kc_middle'].append(kc_middle)
            dict_for_state_df['kc_upper'].append(kc_upper)
            dict_for_state_df['kc_lower'].append(kc_lower)
            upperBB, middleBB, lowerBB = self.instrument_data_dict[instrument].get_bollinger_bands()
            dict_for_state_df['upperBB'].append(upperBB)
            dict_for_state_df['middleBB'].append(middleBB)
            dict_for_state_df['lowerBB'].append(lowerBB)
            
            # self.evt.consoleLog("instrument:", instrument, "upperBB:", upperBB, "; middleBB:", middleBB, "; lowerBB", lowerBB, "; rsi:", rsi, "; kc_middle:", kc_middle, "; kc_upper", kc_upper, "; kc_lower", kc_lower)
        
        df = pd.DataFrame.from_dict(dict_for_state_df)
        self.obs = AlgoEvent._initiate_state(df, self.previous_state, self.initial, self.initial_amount, self.stock_dimension, self.ratio_list)

        # inference
        actions, _ = self.model.predict(self.obs)

        # times scale factor
        actions = actions * self.hmax

        # open or close (= buy/sell in stock), then update the state (obs) array
        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]                   # determine whether buy or sell a stock
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        # convert to 2dp for volume
        self._obs_volumne_to_2dp(self.obs)
        # convert actions to fixed 2dp
        self._action_to_2dp(actions)

        # experimental
        # check account balance between algogene platform and our state numpy array
        # replace our balance with the algogene one
        self.obs[0] = ab['availableBalance']
        self.evt.consoleLog(ab)
        self.evt.consoleLog('obs (first few)', self.obs[:1 + self.stock_dimension * 2])
        self.evt.consoleLog('actions:', actions)

        if(ab['availableBalance']<ab['marginUsed']*0.5):
            
            for index in sell_index:
                sell_num_shares, close_trades = self._sell_stock(self.obs, index, actions[index],0)
    
                # -1 for sell
                actions[index] = sell_num_shares * (-1)
                self.place_order_algogene(-1, self.instrument_list[index],sell_num_shares)
        else:
            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
        
                # sell_cost_pct follows the env_kwarg in training script
                sell_num_shares, close_trades = self._sell_stock(self.obs, index, actions[index],1)
    
                # -1 for sell
                actions[index] = sell_num_shares * (-1)
    
                # close orders with tradeIDs returned
                # self.evt.consoleLog('before place_order_algogene:', "instrument: ", self.instrument_list[index], '; buysell: ', -1, ' volume: ',sell_num_shares, '; askPrice: ', max(0, bd[instrument]["askPrice"]))

           
                self.place_order_algogene(-1, self.instrument_list[index],sell_num_shares)

        if(ab['availableBalance']>250000):
            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
    
                # buy_cost_pct follows the env_kwarg in training script
                buy_num_shares = self._buy_stock(self.obs, index, actions[index], bd)
    
                actions[index] = buy_num_shares
                
                # algogene placeOrder with buy_num_shares
                # self.evt.consoleLog('before place_order_algogene:', "instrument: ", self.instrument_list[index], 'volume: ',buy_num_shares, '; bidPrice: ', max(0, bd[instrument]["bidPrice"]))
                self.place_order_algogene(1, self.instrument_list[index], buy_num_shares)


        
        self.evt.consoleLog('After trading obs (first few):', self.obs[:1 + self.stock_dimension * 2])
        # self.evt.consoleLog(df)

        self.previous_state = self.obs

        # update initial state
        self.initial = False
        
        # reset counter
        self.counter_flag = 0


    @staticmethod
    def _initiate_state(df_day, previous_state, initial, initial_amount, stock_dim, tech_indicator_list):
        '''
        Initiate state for model to inference
        '''
        if initial:
            # For Initial State
            if len(df_day.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [initial_amount]
                    + df_day.close.values.tolist()
                    + [0] * stock_dim                                   # change this to Decimal class for algogene 2dp volume      
                    + sum(
                        [
                            df_day[tech].values.tolist()
                            for tech in tech_indicator_list            # other attributes
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [initial_amount]
                    + [df_day.close]
                    + [0] * stock_dim
                    + sum([[df_day[tech]] for tech in tech_indicator_list], [])
                )


            return state
        else:
            # Using Previous State
            if len(df_day.tic.unique()) > 1:

                # convert the volume to float first for inference
                past_volume = previous_state[(stock_dim + 1) : (stock_dim * 2 + 1)]
                for i, pv in enumerate(past_volume):
                    past_volume[i] = float(pv)

                # for multiple stock
                state = (
                    [previous_state[0]]                # captial in last state
                    + df_day.close.values.tolist()
                    + past_volume
                    + sum(
                        [
                            df_day[tech].values.tolist()
                            for tech in tech_indicator_list
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [previous_state[0]]
                    + [df_day.close]
                    + previous_state[
                        (stock_dim + 1) : (stock_dim * 2 + 1)
                    ]
                    + sum([[df_day[tech]] for tech in tech_indicator_list], [])
                )
        return state


    def _obs_volumne_to_2dp(self, obs):
        for index in range(1 + self.stock_dimension, 1 + self.stock_dimension * 2):
            obs[index] = Decimal(obs[index]).quantize(Decimal('.01'))

    def _action_to_2dp(self, action):
        for i, a in enumerate(action):
            action[i] = Decimal(float(a)).quantize(Decimal('.01'))
        
    # sell stock function from the environment
    # to change the state array
    # TODO: redo this part
    def _sell_stock(self, state, index, action,flag):
        
        def _do_sell_normal():

            temp_close_trades = []
            if(flag==1):
                if state[index + 1] > 0:           # check for missing data (first # of stock)
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # perform sell action based on the sign of the action
                    if float(state[index + self.stock_dimension + 1]) > 0:              # check if you have brought any of the stock
                        # Sell only if current asset is > 0
                        # Sell every stock u have.
    
                        # type: Decmial('.01) for correct estimation
                        sell_num_shares = min(
                            abs(action), state[index + self.stock_dimension + 1]             # compare with the amount of stock you have
                        )
    
                        # search from the open orders, get some until the sum of volume >= sell_num_shares
                        # temp_sell_num_shares = Decimal('0.00')
                        # for trade in self.instrument_openorders[self.instrument_list[index]]:
                        #     temp_sell_num_shares += trade.volume
                        #     temp_close_trades.append(trade)
    
                        #     if temp_sell_num_shares >= sell_num_shares:
                        #         break
    
                        # bid_price = df.loc[df.tic == self.instrument_list[index]]['b'][index]
                        
                        # sell_amount = (
                        #     bid_price
                        #     * sell_num_shares
                        #     * (1 - sell_cost_pct)
                        # )
                        # self.evt.consoleLog('instrument:', self.instrument_list[index], '; bid_price:' , bid_price, '; sell_num_shares:', sell_num_shares, '; sell_amount: ', sell_amount)
                        
                        # update balance
                        # state[0] += sell_amount
    
                        sell_num_shares = Decimal(float(sell_num_shares)).quantize(Decimal('.01'))
                        # state[index + self.stock_dimension + 1] -= sell_num_shares
                    else:
                        sell_num_shares = Decimal('0.00')
                else:
                    sell_num_shares = Decimal('0.00')
            else:
                sell_num_shares =  state[index + self.stock_dimension + 1]             # compare with the amount of stock you have
                sell_num_shares = Decimal(float(sell_num_shares)).quantize(Decimal('.01'))

            return sell_num_shares, temp_close_trades

        # perform sell action based on the sign of the action
        sell_num_shares, temp_close_trades = _do_sell_normal()

        return sell_num_shares, temp_close_trades


    # buy function from the environment
    # to change the state array
    def _buy_stock(self, state, index, action, bd):
        def _do_buy():
            if state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                # divide by the ask_price
                available_amount = Decimal(state[0] / bd[self.instrument_list[index]]['bidPrice']).quantize(Decimal('.01'), rounding=ROUND_DOWN)        # drop any zeros
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)              # buy as much as you can

                # ask_price = df.loc[df.tic == self.instrument_list[index]]['a'][index]
                # buy_amount = (
                #     ask_price * buy_num_shares * (1 + buy_cost_pct)
                # )
                # self.evt.consoleLog('instrument:', self.instrument_list[index], '; ask_price:' , ask_price, '; buy_num_shares:', buy_num_shares, '; buy_amount: ', buy_amount)
                # state[0] -= buy_amount                                 # deduce from capital

                buy_num_shares = Decimal(float(buy_num_shares)).quantize(Decimal('.01'))
                # state[index + self.stock_dimension + 1] += buy_num_shares
            else:
                buy_num_shares = Decimal('0.00')

            return buy_num_shares                                           # buy how many shares

        # perform buy action based on the sign of the action
        buy_num_shares = _do_buy()
        
        return buy_num_shares

    def place_order_algogene(self, buysell, instrument, volume, tradeID=None):
        '''
        Place an order in algogene backtest environment
        instrument: the tic of the instrument
        action: number of stocks u want to buy/sell
        buysell: 1 if buy, -1 if sell
        '''
        
        if abs(volume) == Decimal('0.00'):
            return

        order = AlgoAPIUtil.OrderObject(
            instrument=instrument,
            openclose='open', 
            buysell=1 if buysell >= 0 else -1,                      # always buy call   
            ordertype=0,                    # 0 = market, 1 = limit
            volume=float(abs(volume))
        )

        self.evt.sendOrder(order)


    def on_marketdatafeed(self, md, ab):
        pass

    def on_newsdatafeed(self, nd):
        pass

    def on_weatherdatafeed(self, wd):
        pass
    
    def on_econsdatafeed(self, ed):
        pass
        
    def on_corpAnnouncement(self, ca):
        pass

    def on_orderfeed(self, of):
        trade_status = of.status
        openclose = of.openclose            # 'open' = open a order (buy), 'close' = close a order (sell)
        execution_time = of.insertTime
        tradeID = of.tradeID
        instrument = of.instrument
        buysell = of.buysell
        execution_price = of.fill_price
        execution_volume = of.fill_volume

        # for on_openPositionfeed
        self.on_orderfeed_inst = instrument
        
        self.evt.consoleLog('on_orderfeed: tradeID', tradeID, '; instrument: ', instrument, 'openclose: ', openclose, '; execution_price: ', execution_price, '; execution_volume:', execution_volume)

        # update the volume to state
        if (trade_status == 'success'):
            volume = self.obs[1 + self.stock_dimension + self.instrument_name_to_index_dict[instrument]]
            
            if buysell == 1:
                # add volume
                volume += Decimal(execution_volume).quantize(Decimal('.01'))
            else:
                volume -= Decimal(execution_volume).quantize(Decimal('.01'))

            volume = max(Decimal('0.00'), volume)           # prevent negative values
            self.obs[1 + self.stock_dimension + self.instrument_name_to_index_dict[instrument]] = volume

    def on_dailyPLfeed(self, pl):
        pass

    def on_openPositionfeed(self, op, oo, uo):
        # store all open orders of the instrument just traded
        target = self.on_orderfeed_inst
        trade_array = []

        for tradeID in oo:
            if oo[tradeID]['instrument'] == target:
                # self.evt.consoleLog(oo[tradeID])
                trade = Trade(tradeID, oo[tradeID])
                trade_array.append(trade)

        self.instrument_openorders[target] = trade_array


