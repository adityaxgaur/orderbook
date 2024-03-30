#!/bin/python3

import math
import os
import random
import re
import sys

'''
 Run the matching engine for a list of input operations and returns the trades and orderbooks in a
 csv-like format. Every command starts with either "INSERT", "UPDATE" or "CANCEL" with additional
 data in the columns after the command.

 In case of insert the line will have the format:
 INSERT,<order_id>,<symbol>,<side>,<price>,<volume>
 e.g. INSERT,4,FFLY,BUY,23.45,12

 In case of update the line will have the format:
 UPDATE,<order_id>,<price>,<volume>
 e.g. UPDATE,4,23.12,11

 In case of cancel the line will have the format:
 CANCEL,<order_id>
 e.g. CANCEL,4

 Side will always be "BUY" or "SELL".
 A price is a string with a maximum of 4 digits behind the ".", so "2.1427" and "33.42" would be
 valid prices but "2.14275" would not be a valid price since it has more than 4 digits behind the
 comma.
 A volume will be an integer

 The expected output is:
 - List of trades in chronological order with the format:
   <symbol>,<price>,<volume>,<taker_order_id>,<maker_order_id>
   e.g. FFLY,23.55,11,4,7
   The maker order is the one being removed from the order book, the taker order is the incoming one nmatching it.
 - Then, per symbol (in alphabetical order):
   - separator "===<symbol>==="
   - bid and ask price levels (sorted best to worst by price) for that symbol in the format:
     SELL,<ask_price>,<ask_volume>
     SELL,<ask_price>,<ask_volume>
     BUY,<bid_price>,<bid_volume>
     BUY,<bid_price>,<bid_volume>
     e.g. SELL,25.67,102
          SELL,25.56,34
          BUY,25.52,23
          BUY,25.51,11
          BUY,25.43,4
'''

import time

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        
class BST:
    """ Binary seach tree implementation"""
    def __init__(self):
        self.root = None
        
    def insert(self, value):
        new_node = Node(value)
        
        if self.root is None:
            self.root = new_node
        else:
            current = self.root
            parent = None
            
            while current is not None:
                parent = current
                if value < current.key:
                    current = current.left
                else:
                    current = current.right
                    
            if value < parent.key:
                parent.left = new_node
            else:
                parent.right = new_node
                
    def find_min_node(self, node):
        if node.left is None:
            return node
        return self.find_min_node(node.left)
                
    def delete_node(self, root, value):
        if root is None:
            return root
        
        if value < root.key:
            root.left = self.delete_node(root.left, value)
        elif value > root.key:
            root.right = self.delete_node(root.right, value)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            
            temp = self.find_min_node(root.right)
            root.key = temp.key
            root.right = self.delete_node(root.right, temp.key)
        
        return root
        
    def delete(self, value):
        self.root = self.delete_node(self.root, value)
        
    def minimum(self):
        if self.root is None:
            return None
        node = self.find_min_node(self.root)
        return node.key
        
    def maximum(self):
        if self.root is None:
            return None
        node = self.root
        while node.right:
            node = node.right
        return node.key
        
        
        
    def inorder(self, node) -> list:
        """ returns ascending list : left->root->right"""
        res = []
        if node:
            res = self.inorder(node.left)
            res.append(node.key)
            res = res + self.inorder(node.right)
        return res
        
    def postorder(self, node) -> list:
        """ returns descending order: right->root->left"""
        res = []
        if node:
            res = self.postorder(node.right)
            res.append(node.key)
            res = res + self.postorder(node.left)
        return res
        
    def values(self, reverse=False):
        if reverse:
            return self.postorder(self.root)
        return self.inorder(self.root)

TRADES = []  # ideally below to MatchineEngine class

class OrderNotFoundException(Exception):
    pass

class Order:
    """ represent an order in orderbook"""
    def __init__(self, order_id, symbol=None, side=None, price=None, volume=None, timestamp=None):
        self.order_id = int(order_id)
        self.symbol = symbol
        self.side = side
        if price:
            self._validate_price(price)
            self.price = float(price)
        self._validate_volume(volume)
        self.volume = int(volume) if volume else None
        self.timestamp = timestamp
    
    def _validate_price(self, price):
        if not re.match(r'^\d+(\.\d{1,4})?$', price):
            raise ValueError('Invalid price format. Price should have up to 4 decimal places')
    
    def _validate_volume(self, volume):
        if volume and int(volume) < 0:
            raise ValueError(f'Volume:{volume} should be greater than 0')
        
    def __repr__(self):
        return f"Order({', '.join(f'{key}={value}' for key, value in self.__dict__.items() if value)})"
        
class Trade:
    """Trade done after matching orders"""
    def __init__(self, symbol, price, volume, taker_order_id, maker_order_id):
        self.symbol = symbol
        self.price = price
        self.volume = volume
        self.taker_order_id = taker_order_id
        self.maker_order_id = maker_order_id
        
    def __str__(self):
        return f"{self.symbol},{self.price:g},{self.volume},{self.taker_order_id},{self.maker_order_id}"
    
        
class OrderPrices:
    """Abstact out how the prices are stored inside Orderbook
    """
    def __init__(self, side):
        self.price_bst = BST()  # prices stored in BST
        self.side = side  # BUY or SELL
        self.price_map = {}  # price -> list of orders sorted by timestamp
        self.order_map = {}  # order_id -> order
        self.max_price = None
        self.min_price = None
        
    @property
    def is_buy_side(self) -> bool:
        return self.side == 'BUY'
        
    def order_exists(self, order_id: int) -> bool:
        order = self.order_map.get(order_id)
        return bool(order)
    
    def best_price(self)->float:
        """
        returns max buy price if side == 'BUY'
        and min sell price if side == 'SELL'
        """
        price = self.min_price
        if self.is_buy_side:
            price = self.max_price
        return price
        
    def get_order(self, price):
        """ return the first order from the sorted array. ideally could be a queue"""
        return self.price_map[price][0]
    
    def create_price(self, price: float):
        if self.max_price is None or price > self.max_price:
            self.max_price = price
        if self.min_price is None or price < self.min_price:
            self.min_price = price
            
        self.price_bst.insert(price)
        
    def remove_price(self, price: float):
        self.price_bst.delete(price)
        if self.max_price == price:
            self.max_price = self.price_bst.maximum()
        if self.min_price == price:
            self.min_price = self.price_bst.minimum()
            
    def insert_order(self, order: Order):
        if order.price not in self.price_map:
            self.create_price(order.price)
        self.price_map.setdefault(order.price, []).append(order)
        self.order_map[order.order_id] = order
        
    def update_volume(self, order: Order):
        # Volume changed
        orig_order = self.order_map[order.order_id]
        is_volume_reduced = order.volume < orig_order.volume
        orig_order.volume = order.volume
        if not is_volume_reduced:
            # order list in price_map is sorted by timestamp so update moves order to end of the list
            orig_order.timestamp = order.timestamp
            self.price_map[orig_order.price].remove(orig_order)
            self.price_map[orig_order.price].append(orig_order)
            
    def cancel_order(self, order_id: int):
        orig_order = self.order_map.get(order_id)
        if not orig_order:
            raise OrderNotFoundException(f'Order not found for id:{order_id}')
        orders = self.price_map[orig_order.price]
        orders.remove(orig_order)  # O(n) slow operation !!
        if len(orders) == 0:
            del self.price_map[orig_order.price]
            self.remove_price(orig_order.price)
            
        del self.order_map[order_id]
        
    def price_depth(self) -> list[str]:
        prices = self.price_bst.values(reverse=True)
        result = []
        for price in prices:
            orders = self.price_map[price]
            tot_vol = sum([order.volume for order in orders])
            result.append(','.join([f"{self.side},{price:g},{tot_vol}"]))
        return result
        
class OrderBook:
    """Represent orderbook for a given symbol.
    
    """
    def __init__(self, symbol: str):
        """
        """
        self.symbol = symbol
        self.buy_prices = OrderPrices('BUY')
        self.sell_prices = OrderPrices('SELL')
        self.trades = []
        
    def insert_order(self, order: Order) -> None:
        """ If bxuy order:
            if sell order -> same operation but with sell_prices and sell_prices_map
        """
        # perform order matching
        order = self.perform_order_matching(order)
        if order.volume == 0:
            return
        if order.side == 'BUY':
            self.buy_prices.insert_order(order)
        else:
            self.sell_prices.insert_order(order)
            
    def update_order(self, order: Order) -> None:
        is_buy_side = self.buy_prices.order_exists(order.order_id)
        if is_buy_side:
            orig_order = self.buy_prices.order_map.get(order.order_id)
            if not orig_order:
                raise OrderNotFoundException(f'Order not found for id: {order.order_id}')
            if order.price != orig_order.price:
                # price has changed so cancel the orig order and insert new
                self.buy_prices.cancel_order(orig_order.order_id)
                order.side = orig_order.side
                order.symbol = orig_order.symbol
                self.insert_order(order)
            else:
                self.buy_prices.update_volume(order)
        else:
            orig_order = self.sell_prices.order_map.get(order.order_id)
            if not orig_order:
                raise OrderNotFoundException(f'Order not found for id: {order.order_id}')
            if order.price != orig_order.price:
                # price has changed so cancel the orig order and insert new
                self.sell_prices.cancel_order(orig_order.order_id)
                order.side = orig_order.side
                order.symbol = orig_order.symbol
                self.insert_order(order)
            else:
                self.sell_prices.update_volume(order)
            
    def cancel_order(self, order: Order) -> None:
        is_buy_side = self.buy_prices.order_exists(order.order_id)
        if is_buy_side:
            self.buy_prices.cancel_order(order.order_id)
        else:
            self.sell_prices.cancel_order(order.order_id)
            
            
    def perform_order_matching(self, order: Order):
        "blah"
        global TRADES
        if order.side == 'BUY':
            best_price = self.sell_prices.best_price()
            while best_price is not None and \
                    order.price >= best_price and \
                    order.volume > 0:  # order with 0 qty will not be matched
                
                filled_order = self.sell_prices.get_order(best_price)
                matched_volume = min(order.volume, filled_order.volume)
                trade = Trade(order.symbol, best_price,
                              matched_volume, order.order_id,
                              filled_order.order_id)
                TRADES.append(trade)
                order.volume -= matched_volume
                filled_order.volume -= matched_volume
                if filled_order.volume == 0:
                    # cancel this order as volume = 0
                    self.cancel_order(filled_order)
                best_price = self.sell_prices.best_price()
        else:
            best_price = self.buy_prices.best_price()
            while best_price is not None and \
                    order.price <= best_price and \
                    order.volume > 0:  # order with 0 qty will not be matched

                filled_order = self.buy_prices.get_order(best_price)
                matched_volume = min(order.volume, filled_order.volume)
                trade = Trade(order.symbol, best_price,
                              matched_volume, order.order_id,
                              filled_order.order_id)
                TRADES.append(trade)
                order.volume -= matched_volume
                filled_order.volume -= matched_volume
                if filled_order.volume == 0:
                    # cancel this order as volume = 0
                    self.cancel_order(filled_order)
                best_price = self.buy_prices.best_price()
        # return the remaining order
        return order
    
    def view(self) -> list[str]:
        result = [f"==={self.symbol}==="]
        # print sell orders then buy
        result.extend(self.sell_prices.price_depth())
        result.extend(self.buy_prices.price_depth())
        return result
    
class OrderBookManager:
    """ Maintains multiple order books for different symbols
        
    """
    def __init__(self):
        self.symbol_orderbook_map = {}  # dict(symbol -> order_book)
        self.order_symbol_map = {}  # dict(order_id -> symbol)
        
    def get_order_book(self, symbol: str) -> OrderBook:
        """ returns OrderBook for a give symbol"""
        if symbol not in self.symbol_orderbook_map:
            self.symbol_orderbook_map[symbol] = OrderBook(symbol)
        return self.symbol_orderbook_map[symbol]
        
    def process_order(self, order, action):
        """ delegate INSERT, UPDATE and CANCEL action to orderbook"""
        if action == 'INSERT':
            self.order_symbol_map[order.order_id] = order.symbol
            orderbook = self.get_order_book(order.symbol)
            orderbook.insert_order(order)
        else:
            symbol = self.order_symbol_map.get(order.order_id)
            if not symbol:
                raise OrderNotFoundException(f'Order not found for id: {order.order_id}')
            orderbook = self.get_order_book(symbol)
            if action == 'UPDATE':
                orderbook.update_order(order)
            elif action == 'CANCEL':
                orderbook.cancel_order(order)
        
    def view(self) -> list[str]:
        global TRADES
        result = [str(trade) for trade in TRADES]
        symbols = sorted(self.symbol_orderbook_map.keys())
        for symbol in symbols:
            orderbook = self.symbol_orderbook_map[symbol]
            result.extend(orderbook.view())
        return result
            
                           
#
# Complete the 'runMatchingEngine' function below.
#
# The function is expected to return a STRING_ARRAY.
# The function accepts STRING_ARRAY operations as parameter.
#
def runMatchingEngine(operations: list[str]) -> list[str]:
    global TRADES
    TRADES = []
    order_book_mgr = OrderBookManager()
    for item in operations:
        action, *fields = item.split(',')
        ts = time.time()
        try:
            if action == 'INSERT':
                order_id, symbol, side, price, volume = fields[:5]
                order = Order(order_id=order_id, symbol=symbol, side=side,
                              price=price, volume=volume, timestamp=ts
                              )
            elif action == 'UPDATE':
                order_id, price, volume = fields[:3]
                order = Order(order_id=order_id, price=price, volume=volume, timestamp=ts)
            elif action == 'CANCEL':
                order_id, = fields[0]
                order = Order(order_id=order_id)
            else:
                pass
        except ValueError:
            print(f'Order format is not correct : {item}')
            continue
        else:
            try:
                order_book_mgr.process_order(order, action)
            except OrderNotFoundException:
                print(f'Order does not exist : {item}')
                continue
        
        
    return order_book_mgr.view()
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    operations_count = int(input().strip())
    operations = []

    for _ in range(operations_count):
        operations_item = input()
        operations.append(operations_item)
    
    result = runMatchingEngine(operations)

    fptr.write('\n'.join(result))
    fptr.write('\n')

    fptr.close()
