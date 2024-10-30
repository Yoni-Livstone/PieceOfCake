import os
import pickle
from typing import List
from scipy.optimize import linear_sum_assignment

import numpy as np
import logging

import constants
import random


class Player:
	def __init__(self, rng: np.random.Generator, logger: logging.Logger,
				precomp_dir: str, tolerance: int) -> None:
		"""Initialise the player with the basic information

			Args:
				rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
				logger (logging.Logger): logger use this like logger.info("message")
				precomp_dir (str): Directory path to store/load pre-computation
				tolerance (int): tolerance for the cake distribution
				cake_len (int): Length of the smaller side of the cake
		"""

		self.rng = rng
		self.logger = logger
		self.tolerance = tolerance
		self.cake_len = None
		self.moves = []  # Store the moves made by the player
		
	def which_wall_am_i_on(self, cur_pos, cake_width, cake_len):
		"""
			Function to determine which wall the amoeba is currently on
		"""
		if cur_pos[0] == 0:
			return "left"
		elif cur_pos[0] == cake_width:
			return "right"
		elif cur_pos[1] == 0:
			return "top"
		elif cur_pos[1] == cake_len:
			return "bottom"
		else:
			return "middle"
	

	def move(self, current_percept) -> (int, List[int]):
		polygons = current_percept.polygons
		turn_number = current_percept.turn_number
		cur_pos = current_percept.cur_pos
		requests = current_percept.requests
		cake_len = current_percept.cake_len
		cake_width = current_percept.cake_width

		if turn_number == 1:

			# Initialize and load everything here

			# -----------------------

			# Then...

			first_move = self.moves[0] if self.moves else (0, 0)


			return constants.INIT, first_move[0]
		
		# else while there are moves to be made
		if self.moves:
			next_move = self.moves.pop(0)
			return constants.CUT, next_move
		# else, match the polygons to the requests
		else:
			# Match polygons to requests
			assignment = self.return_matches(polygons, requests)
		
			return constants.ASSIGN, assignment

	def create_cost_matrix(self, polygons, requests):
		"""
			Function to create a cost matrix based on the areas of polygons and requests
		"""
		n = len(polygons)
		m = len(requests)

		cost_matrix = np.full((n, m), np.inf)  # Initialize with infinity

		for i in range(n):
			for j in range(m):

				difference = abs(polygons[i].area - requests[j])

				if difference <= self.tolerance:
					cost_matrix[i][j] = 0		# No penalty
				else:
					cost_matrix[i][j] = (difference / requests[j]) * 100 # Penalty



		return cost_matrix

	def hungarian_algorithm(self, polygons, requests):
		"""
			Function to implement the Hungarian algorithm for optimal assignment
		"""

		cost_matrix = self.create_cost_matrix(polygons, requests)

		# Use the Hungarian method to find the optimal assignment
		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		matches = []

		for i, j in zip(row_ind, col_ind):
			if cost_matrix[i][j] < np.inf:
				penalty = cost_matrix[i][j]

				# THIS RETURNS AN INDEX!!!! MAY NEED TO BE ADJUSTED
				matches.append((i, j, penalty))


		return matches
	
	def return_matches(self, polygons, requests):
		"""
			Function to return the matches
		"""
		matches = self.hungarian_algorithm(polygons, requests)

		# return the indices of the polygons in order of the requests
		assignment = [match[0] for match in matches]

		return assignment

	
	def cost_function(self, polygons, requests):
		"""
			Function to calculate the total cost of the assignment
		"""
		matches = self.hungarian_algorithm(polygons, requests)
		total_cost = sum(penalty for _, _, penalty in matches)
		return total_cost
	
	def hill_climbing(self, polygons, requests, cuts):
		"""
			Function to implement hill climbing optimization for assignment
		"""
		current_objective = self.cost_function(polygons, requests)

		current_cuts = cuts.copy()

		while True:
			# Get the neighborhood of the current cuts
			neighborhood = self.get_neighborhood(current_cuts)

			# Try moving cuts

			improved = False
			for neighbor in neighborhood:
				neighbor_objective = self.cost_function(polygons, requests)

				if neighbor_objective < current_objective:
					current_cuts = neighbor
					current_objective = neighbor_objective
					improved = True

			if not improved:
				break

			# Try adding cuts
			if not improved:
				# Implement logic to add cuts
				pass

		return current_cuts
	
	def get_neighborhood(self, cuts):
		"""
			Function to get the neighborhood of the current cuts
		"""
		neighborhood = []
		# Implement your logic here
		return neighborhood




	def translate_to_valid_moves(self, cuts : List[List[tuple]]) -> int, [int, int]:
		"""
			Function to translate the matches to valid moves

			It should use the breadcrumb trail to determine the path to take
		"""
		pass
		# Pick the starting location
		moves = []

		# Pick location
		cur_location = cuts[0][0]  # Start from the first cut

		next_cut = cuts[0][1]  # Next cut



		self.add_next_move(moves, cur_location)

		# for each list in list
		for cut in cuts:						# each cut is a list of 2 tuples
			for move in cut:
				# translate move to valid move
				pass

	def traverse_to_next_cut(self, current_pos, next_cut):
		"""
			Function to traverse to the next cut
		"""
		pass

		moves = []

		# destination is the eldge of the cut closest to current_pos
		# CHECK THIS LOGIC
		dest = next_cut[0] if current_pos[0] < next_cut[0] else next_cut[1]

		# if I am at a corner
		if (current_pos[0] == 0 and current_pos[1] == 0) or (current_pos[0] == 0 and current_pos[1] == self.cake_len) or (current_pos[0] == self.cake_width and current_pos[1] == 0) or (current_pos[0] == self.cake_width and current_pos[1] == self.cake_len):
			# Implement your logic here
			pass

		# if I am on the North Wall or South Wall
		elif current_pos[1] == 0 or current_pos[1] == self.cake_len:
			# Go to the wall closest to destination

			pass

		# if I am on the East Wall or West Wall
		elif current_pos[0] == 0 or current_pos[0] == self.cake_width:
			# Implement your logic here
			pass


		# Implement your logic here
		return next_pos
	
	def add_next_move(self, moves, next_pos):
		"""
			Function to add the next move to the moves list
		"""
		pass

		cur_pos = moves[-1] if moves else None

		if cur_pos is None:
			moves.append(next_pos)
		else:

		
		# Implement your logic here
		return moves

	def find_next_point(self, current_pos, next_cut, cuts):
		"""
			Function to find the next point to move to
		"""
		pass

		# Implement your logic here
		return next_pos

	def cut(self, next_pos):
		"""
			Function to perform the cut
		"""

		return constants.CUT, next_pos

	def add_move(self, moves : List[List[tuple]], move : tuple):
		"""
			Function to add a move to the moves list
		"""

		last_vector = moves[-1] 

		last_location = last_vector[-1]

		next_vector = []

		next_vector.append((last_location, move))

		moves.append(next_vector)

		return moves
	
	def find_next_point(self, current_pos : tuple, cuts : List[List[tuple]]):
		"""
			Function to find the next point to move to
		"""
		pass

		# find the tuple in cuts that is closest to current_pos
		locations = []

		for cut in cuts:
			for point in cut:
				locations.append(point)

		# find closest point to tuple
		closest = min(locations, key=lambda x: np.linalg.norm(np.array(x) - np.array(current_pos)))

		return closest
		


# N = 100000
# Cuts = []


# For I in range (n):
#     if I %2:
#         // left wall:
#         cuts.append(current location, random destination) 


# -----------------

# Score = infinity

# While score is still decreasing:

#     for 




# Find initial starting point

# Make first cut

# Traverse to next start point

# Make next cut







def find_closest_vector(current_pos : tuple[int, int], cuts : List[List[tuple]]):
	"""
		Function to find the closest vector to the current position
	"""

	closest = None
	closest_distance = float('inf')

	for cut in cuts:
		for point in cut:
			distance = np.linalg.norm(np.array(current_pos) - np.array(point))
			if distance < closest_distance:
				closest_distance = distance
				closest = point

	return closest

def traverse_with_breadcrumbs(start_pos : tuple[int, int], end_pos : tuple[int, int], cake_len: int, cake_width: int) -> List[tuple[int, int]]:
	"""
		Function to traverse from start_pos to end_pos using breadcrumbs
	"""

	top_left_counter = 0.00
	bottom_left_counter = 0.00
	top_right_counter = 0.00
	bottom_right_counter = 0.00



	path = []


	# Determine current edge position
	if start_pos[0] == 0:  # Left wall
		
		


	

	# Implement your logic here
	return path


def classify_position(self, start_point, end_point):
        """
        Classifies the position of the end_point relative to the start_point on the cake's edge.
        
        Returns:
            0: Same edge
            1: Adjacent edge
            2: Opposite edge
            -1: Start point is not on any edge
        """
        x1, y1 = 0, 0
        x2, y2 = self.cake_len, self.cake_width

        x, y = start_point
        dx, dy = end_point

        # Check if start_point is on the left edge
        if x == x1 and y1 <= y <= y2:
            if dx == x1 and y1 <= dy <= y2:
                return 0  # Same left edge
            elif dy < y1 or dy > y2:
                return 1  # Adjacent edge
            elif dx == x2 and y1 <= dy <= y2:
                return 2  # Opposite right edge

        # Check if start_point is on the right edge
        elif x == x2 and y1 <= y <= y2:
            if dx == x2 and y1 <= dy <= y2:
                return 0  # Same right edge
            elif dy < y1 or dy > y2:
                return 1  # Adjacent edge
            elif dx == x1 and y1 <= dy <= y2:
                return 2  # Opposite left edge

        # Check if start_point is on the top edge
        elif y == y1 and x1 <= x <= x2:
            if dy == y1 and x1 <= dx <= x2:
                return 0  # Same top edge
            elif dx < x1 or dx > x2:
                return 1  # Adjacent edge
            elif dy == y2 and x1 <= dx <= x2:
                return 2  # Opposite bottom edge

        # Check if start_point is on the bottom edge
        elif y == y2 and x1 <= x <= x2:
            if dy == y2 and x1 <= dx <= x2:
                return 0  # Same bottom edge
            elif dx < x1 or dx > x2:
                return 1  # Adjacent edge
            elif dy == y1 and x1 <= dx <= x2:
                return 2  # Opposite top edge

        return -1  # Point is not on any edge



def generate_random_cuts(length, width, n):
	cuts = set()
    
	while len(cuts) < n:
        # Randomly select the starting wall
		start_wall = random.choice(['left', 'right', 'top', 'bottom'])
        
		if start_wall == 'left':
            # From left wall to either right wall or top/bottom wall
			end_wall = random.choice(['right', 'top', 'bottom'])
			x_start = 0  # Left wall
			y_start = round(random.uniform(0, length), 2)  # Random y-coordinate
            
			if end_wall == 'right':
				x_end, y_end = width, y_start
			elif end_wall == 'top':
				x_end, y_end = round(random.uniform(0, width), 2), 0  # Top wall
			else:  # end_wall == 'bottom'
				x_end, y_end = round(random.uniform(0, width), 2), length  # Bottom wall

		elif start_wall == 'right':
            # From right wall to either left wall or top/bottom wall
			end_wall = random.choice(['left', 'top', 'bottom'])
			x_start = width  # Right wall
			y_start = round(random.uniform(0, length), 2)  # Random y-coordinate
            
			if end_wall == 'left':
				x_end, y_end = 0, y_start
			elif end_wall == 'top':
				x_end, y_end = round(random.uniform(0, width), 2), 0  # Top wall
			else:  # end_wall == 'bottom'
				x_end, y_end = round(random.uniform(0, width), 2), length  # Bottom wall

		elif start_wall == 'top':
            # From top wall to either bottom wall or left/right wall
			end_wall = random.choice(['bottom', 'left', 'right'])
			x_start = round(random.uniform(0, width), 2)  # Random x-coordinate
			y_start = 0  # Top wall
            
			if end_wall == 'bottom':
				x_end, y_end = x_start, length  # Bottom wall
			elif end_wall == 'left':
				x_end, y_end = 0, round(random.uniform(0, length), 2)  # Left wall
			else:  # end_wall == 'right'
				x_end, y_end = width, round(random.uniform(0, length), 2)  # Right wall

		else:  # start_wall == 'bottom'
            # From bottom wall to either top wall or left/right wall
			end_wall = random.choice(['top', 'left', 'right'])
			x_start = round(random.uniform(0, width), 2)  # Random x-coordinate
			y_start = length  # Bottom wall
            
			if end_wall == 'top':
				x_end, y_end = x_start, 0  # Top wall
			elif end_wall == 'left':
				x_end, y_end = 0, round(random.uniform(0, length), 2)  # Left wall
			else:  # end_wall == 'right'
				x_end, y_end = width, round(random.uniform(0, length), 2)  # Right wall

        # Create a cut vector and add to the set
		cut = [(round(x_start, 2), round(y_start, 2)), (round(x_end, 2), round(y_end, 2))]
		cuts.add(tuple(map(tuple, cut)))  # Convert to tuple of tuples for set

    # Convert back to the desired list of lists format
	return [list(map(list, cut)) for cut in cuts]  # Convert back to list of lists



# Simulate cuts

from shapely.geometry import Polygon, LineString
from shapely.ops import split

def simulate_cuts(cuts, cake_len, cake_width):
	
	initial_rectangle = Polygon([(0, 0), (self.cake_len, 0), (self.cake_width, self.cake_length), (0, self.cake_length)])


	# EXAMPLE FORMAT FOR CUTS
	# cuts = [ [(1, 0), (1, 10)],   # Vertical cut at x = 1
    #      [(0, 5), (5, 5)],   # Horizontal cut at y = 5
    #      [(3, 0), (3, 10)] ]  # Vertical cut at x = 3

	polygon_list = [initial_rectangle]

	for cut in cuts:
		line_segment = LineString(cut)
		new_polygons = []
		for polygon in polygon_list:
			if line_segment.intersects(polygon):
				# Split the polygon
				result = split(polygon, line_segment)
				new_polygons.extend(result.geoms)
			else:
				new_polygons.append(polygon)
		polygon_list = new_polygons  # Update the polygon list

	areas = [poly.area for poly in polygon_list]

	return areas


def simulate_cuts_2(cuts, cake_len, cake_width):
	"""
		Function to simulate cuts on the cake and return the areas of the resulting pieces.
	"""
	# Create the initial cake as a polygon
	cake = Polygon([(0, 0), (cake_width, 0), (cake_width, cake_len), (0, cake_len)])

	newPieces = []
	for polygon in polygon_list:
		line_points = LineString([tuple(cur_pos), tuple(action[1])])
		slices = divide_polygon(polygon, line_points)
		for slice in slices:
			newPieces.append(slice)

	polygon_list = newPieces


	areas = [poly.area for poly in polygon_list]

	return areas

def divide_polygon(self, polygon, line):
        """
        Divide a convex polygon by a line segment into two polygons.

        Parameters:
        - polygon: A convex polygon (as a Shapely Polygon object)
        - line_points: A list containing two points that represent the line segment

        Returns:
        - Two polygons (as shapely Polygon objects) that result from dividing the original polygon
        """
        # Create the convex polygon and the line segment using Shapely
        # polygon = Polygon(polygon_points)
        # line = LineString(line_points)

        # Check if the line intersects with the polygon
        if not line.intersects(polygon):
            return [polygon]
        # Split the polygon into two pieces
        result = split(polygon, line)

        # Convert the result into individual polygons

        polygons = []
        for i in range(len(result.geoms)):
            polygons.append(result.geoms[i])

        return polygons
