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
        
    def values(self, reverse=False) -> None:
        if reverse:
            return self.postorder(self.root)
        return self.inorder(self.root)

TRADES = []  # Global variable to store list of trades done.

class OrderNotFoundException(Exception):
    pass

class Order:
    """ BUY or SELL order."""
    def __init__(self, order_id, symbol=None, side=None, price=None, volume=None) -> None:
        self.order_id = int(order_id)
        self.symbol = symbol
        self.side = side
        if price:
            self._validate_price(price)
            self.price = float(price)
        self._validate_volume(volume)
        self.volume = int(volume) if volume else None
        self.timestamp = time.time()
    
    def _validate_price(self, price) -> None:
        """ price should have up to 4 decimal places"""
        if not re.match(r'^\d+(\.\d{1,4})?$', price):
            raise ValueError('Invalid price format. Price should have up to 4 decimal places')
    
    def _validate_volume(self, volume) -> None:
        if volume and int(volume) < 0:
            raise ValueError(f'Volume:{volume} should be greater than 0')
        
    def __repr__(self) -> str:
        return f"Order({', '.join(f'{key}={value}' for key, value in self.__dict__.items() if value)})"
        
class Trade:
    """Trade to fill order after matching"""
    def __init__(self, symbol, price, volume, taker_order_id, maker_order_id) -> None:
        self.symbol = symbol
        self.price = price  # best bid or ask price available from market maker order
        self.volume = volume
        self.taker_order_id = taker_order_id  # new or update order
        self.maker_order_id = maker_order_id  # existing order in orderbook which can match taker order
        
    def __str__(self) -> None:
        return f"{self.symbol},{self.price:g},{self.volume},{self.taker_order_id},{self.maker_order_id}"
    
        
class OrderPrices:
    """
        OrderPrices manages prices using a binary search tree (BST).
        Each price is associated with a list of orders, sorted by timestamp, and stored in a dictionary.
        Additionally, individual orders are stored in a dictionary keyed by their unique order ID.
    """
    def __init__(self, side):
        """
            Change the implementation to store list of order_ids in price_map!!!
        """
        self.price_bst = BST()  # Prices stored in a binary search tree.
        self.side = side  # BUY or SELL order
        self.price_map = {}  # Maps prices to lists of orders sorted by timestamp.
        self.order_map = {}  # Maps order IDs to individual orders.
        self.max_price = None  # Highest price on the buy(bid) side
        self.min_price = None  # Lowest price on the sell(ask) side
        
    @property
    def is_buy_side(self) -> bool:
        """
        Checks if the current side is for buying
        """
        return self.side == 'BUY'
        
    def order_exists(self, order_id: int) -> bool:
        """
        Checks if an order with give ID exists
        """
        order = self.order_map.get(order_id)
        return bool(order)
    
    def best_price(self) -> float:
        """
        Returns the highest buy price if side is 'BUY'(BID)
        or the lowest sell price if side is 'SELL'(ASK)
        """
        price = self.min_price
        if self.is_buy_side:
            price = self.max_price
        return price
        
    def get_order(self, price) -> float:
        """
            Returns the first order from the list of orders sorted by timestamp.
            First order has the highest time priority
        """
        return self.price_map[price][0]
    
    def create_price(self, price: float) -> None:
        """Crates a new price entry in binary search tree"""
        if self.max_price is None or price > self.max_price:
            self.max_price = price
        if self.min_price is None or price < self.min_price:
            self.min_price = price
        self.price_bst.insert(price)
        
    def remove_price(self, price: float) -> float:
        """ Removes the price entry from BST """
        self.price_bst.delete(price)
        if self.max_price == price:
            self.max_price = self.price_bst.maximum()
        if self.min_price == price:
            self.min_price = self.price_bst.minimum()
            
    def insert_order(self, order: Order) -> None:
        """
            Insert order price in BST if doesn't exist already
            Insert Order in sorted order list keyed against the price
            Insert in order_map dict
        """
        if order.price not in self.price_map:
            self.create_price(order.price)
        self.price_map.setdefault(order.price, []).append(order)
        self.order_map[order.order_id] = order
        
    def update_volume(self, order: Order) -> None:
        """ Updates volume of an existing order """
        orig_order = self.order_map[order.order_id]
        is_volume_reduced = order.volume < orig_order.volume
        orig_order.volume = order.volume
        if not is_volume_reduced:
            # If volume is not reduced then order looses it time priority and goes to the end of order list
            orig_order.timestamp = order.timestamp
            orderlist = self.price_map[orig_order.price]
            orderlist.remove(orig_order)
            orderlist.append(orig_order)
            
    def cancel_order(self, order_id: int) -> None:
        """ Cancels an order """
        orig_order = self.order_map.pop(order_id, None)
        if not orig_order:
            raise OrderNotFoundException(f'Order not found for id:{order_id}')
        price_orders = self.price_map[orig_order.price]
        if price_orders:
            price_orders.remove(orig_order)
            if not price_orders:  # if no orders left at this price then remove from price_map and price_bst
                del self.price_map[orig_order.price]
                self.remove_price(orig_order.price)
        
    def price_depth(self) -> list[str]:
        """Retrieves prices from BST in reverse order along with total volume of orders at each level"""
        prices = self.price_bst.values(reverse=True)
        result = []
        for price in prices:
            orders = self.price_map[price]
            tot_vol = sum([order.volume for order in orders])
            result.append(','.join([f"{self.side},{price:g},{tot_vol}"]))
        return result
        
class OrderBook:
    """Represents an order book for a given symbol, containing buy and sell orders.
    Attributes:
        symbol (str) : The symbol associated with the order book
        buy_prices (OrderPrices) : An OrderPrices object managing BUY orders
        sell_prices (OrderPrices) : An OrderPrices object managing SELL orders
        trades(list): A list to store filled orders
    
    """
    def __init__(self, symbol: str):
        """
        Initialize OrderBook for a given symbol
        """
        self.symbol = symbol
        self.buy_prices = OrderPrices('BUY')
        self.sell_prices = OrderPrices('SELL')
        self.trades = []
        
    def insert_order(self, order: Order) -> None:
        """ Perform order matching for a new incoming order.
            Remaining order is added to either buy_prices or sell_prices
        """
        # perform order matching
        remaining_order = self.perform_order_matching(order)
        if remaining_order.volume == 0:
            # order is fully-filled
            return
        if remaining_order.side == 'BUY':
            self.buy_prices.insert_order(remaining_order)
        else:
            self.sell_prices.insert_order(remaining_order)
            
    def update_order(self, order: Order) -> None:
        """ Update an existing order in the order book
        Raises:
           OrderNotFoundException: if order ID is not found in both  
        """
        # Determind whether order is on the buy side or sell side
        is_buy_side = self.buy_prices.order_exists(order.order_id)

        if is_buy_side:
            orig_order = self.buy_prices.order_map.get(order.order_id)
        else:
            orig_order = self.sell_prices.order_map.get(order.order_id)
            
        if not orig_order:
            raise OrderNotFoundException(f'Order not found for id: {order.order_id}')
            
        if is_buy_side:
            if order.price != orig_order.price:
                # price has changed so cancel the orig order and insert new
                self.buy_prices.cancel_order(orig_order.order_id)
                
                # Update the side and symbol of incoming order
                order.side = orig_order.side
                order.symbol = orig_order.symbol
                self.insert_order(order)
            else:
                # price has not changed, update volume of original order
                self.buy_prices.update_volume(order)
        else:
            if order.price != orig_order.price:
                # price has changed so cancel the orig order and insert new
                self.sell_prices.cancel_order(orig_order.order_id)
                
                # Update the side and symbol of incoming order
                order.side = orig_order.side
                order.symbol = orig_order.symbol
                self.insert_order(order)
            else:
                # price has not changed, update volume of original order
                self.sell_prices.update_volume(order)
            
    def cancel_order(self, order: Order) -> None:
        is_buy_side = self.buy_prices.order_exists(order.order_id)
        if is_buy_side:
            self.buy_prices.cancel_order(order.order_id)
        else:
            self.sell_prices.cancel_order(order.order_id)
            
            
    def perform_order_matching(self, order: Order) -> Order:
        """Matches an incoming order with existing order in order book
        Args:
            order (Order): The incoming order to be matched
        Returns:
            Order: The remaining unmatched portion of the order
        """
        global TRADES   # global list to store filled orders from all order books
        
        if order.side == 'BUY':
            # for incoming buy order get the best ask price
            best_price = self.sell_prices.best_price()
            # continously check the best price until either the order is fully filled
            # or no sell order exist in orderbook below the bid price
            while best_price is not None and \
                    order.price >= best_price and \
                    order.volume > 0:  # order with 0 qty will not be matched
                
                filled_order = self.sell_prices.get_order(best_price)
                matched_volume = min(order.volume, filled_order.volume)
                trade = Trade(order.symbol, best_price,
                              matched_volume, order.order_id,
                              filled_order.order_id)
                TRADES.append(trade)  # record filled order
                order.volume -= matched_volume
                filled_order.volume -= matched_volume
                if filled_order.volume == 0:
                    # cancel this order as volume = 0
                    self.cancel_order(filled_order)
                # update the best price for next iteration
                best_price = self.sell_prices.best_price()
        else:
            # for incoming sell order get the best ask price
            best_price = self.buy_prices.best_price()
            # continously check the best price until either the order is fully filled
            # or no buy order exist in orderbook above the ask price
            while best_price is not None and \
                    order.price <= best_price and \
                    order.volume > 0:  # order with 0 qty will not be matched

                filled_order = self.buy_prices.get_order(best_price)
                matched_volume = min(order.volume, filled_order.volume)
                trade = Trade(order.symbol, best_price,
                              matched_volume, order.order_id,
                              filled_order.order_id)
                TRADES.append(trade)  # record filled order
                order.volume -= matched_volume
                filled_order.volume -= matched_volume
                if filled_order.volume == 0:
                    # cancel this order as volume = 0
                    self.cancel_order(filled_order)
                best_price = self.buy_prices.best_price()
        # return the remaining unmatched portion of the incoming order
        return order
    
    def view(self) -> list[str]:
        result = [f"==={self.symbol}==="]
        # print sell orders then buy
        result.extend(self.sell_prices.price_depth())
        result.extend(self.buy_prices.price_depth())
        return result
    
class OrderBookManager:
    """ Manages multiple order books for different symbols
    Attributes
        symbol_orderbook_map (dict) : A dict mapping different symbols to their respective order books
        order_symbol_map (dict): A dict mapping order_id to their symbols
        
    """
    def __init__(self) -> None:
        self.symbol_orderbook_map = {}  # dict(symbol -> order_book)
        self.order_symbol_map = {}  # dict(order_id -> symbol)
        
    def get_order_book(self, symbol: str) -> OrderBook:
        """ Retrieves instance of OrderBook for a given symbol"""
        return self.symbol_orderbook_map.setdefault(symbol, OrderBook(symbol)) 
        
    def process_order(self, order: Order, action: str) -> None:
        """ Delegate INSERT, UPDATE and CANCEL action to appropriate order book
        Args:
            order (Order): Incoming order
            action (str): INSERT, UPDATE or CANCEL 
        Raises:
            OrderNotFoundException : If the order ID is not found in the order_symnol_map
        """
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
        """ Generates a view of order books and the trades"""
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
    """ Executes the order matching operations and returns the resulting order book views
    Args:
        operations(list[str]): A list of string representing incoming orders
    
    Returns
        list[str]: list of strings representing trades done in chronological order followed 
                   by price-depth of all orderbooks
    """
    order_book_mgr = OrderBookManager()
    for item in operations:
        action, *fields = item.split(',')
        try:
            if action == 'INSERT':
                order_id, symbol, side, price, volume = fields[:5]
                order = Order(order_id=order_id, symbol=symbol, side=side,
                              price=price, volume=volume,
                              )
            elif action == 'UPDATE':
                order_id, price, volume = fields[:3]
                order = Order(order_id=order_id, price=price, volume=volume)
            elif action == 'CANCEL':
                order_id, = fields[0]
                order = Order(order_id=order_id)
            else:
                pass  # skip invalid actions 
        except ValueError:
            print(f'Order format is not correct : {item}')
            continue
        else:
            # Process the current order using OrderBookManager 
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
