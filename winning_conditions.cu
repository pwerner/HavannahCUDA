/*	Havannah CUDA – This program provides a CUDA based playout strategy for Monte-Carlo-Evaluation
 *	Copyright (C) 2012  Peter Werner
 *	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *	Contact: Peter@wernerbrothers.de
 */

#include "winning_conditions.h"

bool ringWin(small_board * b, idx mv, const code_board * code) {
	byte color = onMove(b);
	if(checkForRing(b,mv,code,color)) {
		setWinner(b, color);
		setReason(b, RING);
		// for result only
		perf_move(b,mv,code);
		return true;
	}
	return false;
}

// ring checking requires board BEFORE setting move
bool checkForRing(small_board * b, idx move, const code_board * code, byte color) {
	neighbours n;
	getNeighbours(code, move, &n);
	short i;
	short count = 0;
	for(i = 0; i < 6; i++) {
		idx new_neighbour = n.neighbour[i];
		if(new_neighbour != NO_NEIGHBOUR && b->cell[new_neighbour] == color) {
			count++;
			// test whether neighbour now has 6 friendly neighbours
			if(checkForNearRing(b,code,new_neighbour,color)) {
				return true;
			}
		} else {
			n.neighbour[i] = NO_NEIGHBOUR;
		}
	}
	if(count < 2) {
		return false;
	}

	if(n.neighbour[0] != NO_NEIGHBOUR) {
		idx root1 = findRoot(b, n.neighbour[0]);
		idx root2;
		if(n.neighbour[1] != NO_NEIGHBOUR) {
			if((n.neighbour[4] == NO_NEIGHBOUR || n.neighbour[5] == NO_NEIGHBOUR) && (n.neighbour[2] == NO_NEIGHBOUR || n.neighbour[3] == NO_NEIGHBOUR)) {
				root2 = findRoot(b, n.neighbour[1]);
				if(root1 == root2) {
					return true;
				}				
			}
		}
		if(n.neighbour[3] != NO_NEIGHBOUR) {
			if(n.neighbour[2] == NO_NEIGHBOUR && (n.neighbour[4] == NO_NEIGHBOUR||n.neighbour[5] == NO_NEIGHBOUR ||n.neighbour[1] == NO_NEIGHBOUR)) {
				root2 = findRoot(b,n.neighbour[3]);
				if(root1 == root2) {
					return true;
				}	
			}
		}
		if(n.neighbour[5] != NO_NEIGHBOUR) {
			if(n.neighbour[4] == NO_NEIGHBOUR&& (n.neighbour[3] == NO_NEIGHBOUR||n.neighbour[2] == NO_NEIGHBOUR ||n.neighbour[1] == NO_NEIGHBOUR)) {
				root2 = findRoot(b,n.neighbour[5]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	if(n.neighbour[1] != NO_NEIGHBOUR) {
		idx root1 = findRoot(b, n.neighbour[1]);
		idx root2;
		if(n.neighbour[2] != NO_NEIGHBOUR) {
			if(n.neighbour[3] == NO_NEIGHBOUR&& (n.neighbour[4] == NO_NEIGHBOUR||n.neighbour[5] == NO_NEIGHBOUR ||n.neighbour[0] == NO_NEIGHBOUR)) {
				root2 = findRoot(b, n.neighbour[2]);
				if(root1 == root2) {
					return true;
				}
			}
		}
		if(n.neighbour[4] != NO_NEIGHBOUR) {
			if(n.neighbour[5] == NO_NEIGHBOUR&& (n.neighbour[0] == NO_NEIGHBOUR||n.neighbour[2] == NO_NEIGHBOUR ||n.neighbour[3] == NO_NEIGHBOUR)) {
				root2 = findRoot(b, n.neighbour[4]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	if(n.neighbour[2] != NO_NEIGHBOUR) {
		idx root1 = findRoot(b, n.neighbour[2]);
		idx root2;
		if(n.neighbour[4] != NO_NEIGHBOUR) {
			if(n.neighbour[0] == NO_NEIGHBOUR&& (n.neighbour[3] == NO_NEIGHBOUR||n.neighbour[5] == NO_NEIGHBOUR ||n.neighbour[1] == NO_NEIGHBOUR)) {
				root2 = findRoot(b, n.neighbour[4]);
				if(root1 == root2) {
					return true;
				}
			}
		}
		if(n.neighbour[5] != NO_NEIGHBOUR) {
			if((n.neighbour[4] == NO_NEIGHBOUR || n.neighbour[0] == NO_NEIGHBOUR) && (n.neighbour[1] == NO_NEIGHBOUR || n.neighbour[3] == NO_NEIGHBOUR)) {
				root2 = findRoot(b, n.neighbour[5]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	if(n.neighbour[3] != NO_NEIGHBOUR) {
		idx root1 = findRoot(b, n.neighbour[3]);
		idx root2;
		if(n.neighbour[4] != NO_NEIGHBOUR) {
			if((n.neighbour[2] == NO_NEIGHBOUR || n.neighbour[0] == NO_NEIGHBOUR) && (n.neighbour[1] == NO_NEIGHBOUR || n.neighbour[5] == NO_NEIGHBOUR)) {
				root2 = findRoot(b, n.neighbour[4]);
				if(root1 == root2) {
					return true;
				}
			}
		}
		if(n.neighbour[5] != NO_NEIGHBOUR) {
			if(n.neighbour[1] == NO_NEIGHBOUR&& (n.neighbour[4] == NO_NEIGHBOUR||n.neighbour[2] == NO_NEIGHBOUR ||n.neighbour[0] == NO_NEIGHBOUR)) {
				root2 = findRoot(b, n.neighbour[5]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	return false;
}

// checks whether there is a bridge for current color
bool bridgeWin(small_board * b, const code_board * code) {
	// move is done, so check for next = prev player
	byte color = next(b);
	if(checkForBridge(b,code,color)) {
		setWinner(b, color);
		setReason(b, BRIDGE);
		return true;
	}
	return false;
}

// checks whether there is a bridge for current color
bool checkForBridge(small_board * b, const code_board * code, byte colorOnMove) {
	if(b->cell[code->corners[0]] == colorOnMove) {
		idx root1 = findRoot(b, code->corners[0]);
		idx root2;
		if(b->cell[code->corners[1]] == colorOnMove) {
			root2 = findRoot(b, code->corners[1]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[2]] == colorOnMove) {
			root2 = findRoot(b, code->corners[2]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[3]] == colorOnMove) {
			root2 = findRoot(b, code->corners[3]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[1]] == colorOnMove) {
		idx root1 = findRoot(b, code->corners[1]);
		idx root2;
		if(b->cell[code->corners[2]] == colorOnMove) {
			root2 = findRoot(b, code->corners[2]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[3]] == colorOnMove) {
			root2 = findRoot(b, code->corners[3]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[2]] == colorOnMove) {
		idx root1 = findRoot(b, code->corners[2]);
		idx root2;
		if(b->cell[code->corners[3]] == colorOnMove) {
			root2 = findRoot(b, code->corners[3]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[3]] == colorOnMove) {
		idx root1 = findRoot(b, code->corners[3]);
		idx root2;
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[4]] == colorOnMove) {
		idx root1 = findRoot(b, code->corners[4]);
		idx root2;
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	return false;
}


// checks whether there is a fork for current color
bool forkWin(small_board * b, const code_board * code) {
	// move is done, so check for next = prev player
	byte color = next(b);
	if(checkForFork(b,code,color)) {
		setWinner(b, color);
		setReason(b, FORK);
		return true;
	}
	return false;
}

// checks whether there is a fork for current color
bool checkForFork(small_board * b, const code_board * code, byte colorOnMove) {
	int i;
	int j;
	// for 4 edges (if there are three edges connected it will detected within this 4 edges)
	for(i=0; i<4;i++) {
		edge ce = code->edges[i];
		for(j=0; j<size(b)-2;j++) {
			// find friendly cell
			if(b->cell[ce.edge[j]] == colorOnMove) {
				// test for all edges if the cell is at least connected to 3 other edges
				if(isConnectedToThreeEdges(ce.edge[j], i, b, colorOnMove, code)) {
					return true;
				}
			}
		}
	}
	return false;
}

bool isConnectedToThreeEdges(idx index, short orginEdge, small_board * b, byte colorOnMove,const code_board * code) {
	byte count = 0;
	int i;
	// test all other edges besides origin edge
	for(i=0; i < 6; i++) {
		if(i != orginEdge) {
			if(isConnectedToEdge(index,code->edges[i],b,colorOnMove)) {
				count++;
			}
		}
	}
	if(count >= 3) {
		return true;
	}
	return false;
}
	
bool isConnectedToEdge(idx index, edge edge, small_board * b, byte colorOnMove) {
	idx root1 = findRoot(b, index);
	int k;
	for(k = 0; k < size(b)-2; k++) {
		if(b->cell[edge.edge[k]] == colorOnMove) {
			idx root2 = findRoot(b, edge.edge[k]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	return false;
}

bool checkForNearRing(small_board * b,const code_board * code,idx nb, byte color) {
	neighbours nn;
	getNeighbours(code, nb, &nn);
	short j;
	short nc = 0;
	for(j = 0; j < 6; j++) {
		if(nn.neighbour[j] != NO_NEIGHBOUR && b->cell[nn.neighbour[j]] == color) {
			nc++;
		}
	}
	// move not set yet so 5 are enough
	if(nc == 5) {
		return true;
	}
	return false;
}


// does move
void perf_move(small_board * b, idx move, const code_board * code) {
	if(b->time <= ELEMENT_SIZE) {
		// get color for this move
		byte color = onMove(b);
		b->last = move;
		b->cell[move] = color;
		makeSet(b,move);
		neighbours n;
		getNeighbours(code, move, &n);
		short i;
		// just friendly neighours and not off the board
		for(i = 0; i < 6; i++) {
			if(n.neighbour[i] != NO_NEIGHBOUR && b->cell[n.neighbour[i]] == color) {
				unionSets(b, move, n.neighbour[i]);
			}
		}		
		b->time = b->time+1;	
		setNext(b,color);
	} else {
		// board is full ! should not happen
	}
}