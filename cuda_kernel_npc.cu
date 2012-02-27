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

#include "cuda_kernel_npc.h"
#include "code.h"

// execute parallel playout with only MAX_ACTIVE games on device mem
__global__ void playouts_npc_d(const small_board * org_b, game * games, curandState * rng) {
	int id = get_thread_id_npc();
	// create board code
	__shared__ code_board_10 code;
	init_npc_d(&code);
	// get active game
	game * g = &games[id];
	if(g->move == -1) {
		return;
	}
	__shared__ small_board sb;
	small_board * b = &sb;
	// get copy of orignal board
	copy_board_npc_d(org_b,b);
	// perform our move
	perf_move_npc_d(b,g->move,&code);
	// perform playout
	if(playout_npc_d(b,&code,&rng[id])) {
		// loss, since we moved first
		loss_npc_d(g);		
	} else {
		byte re = reason_npc_d(b);
		// could be a draw
		if(NONE == re || b->time == ELEMENT_SIZE) {
			draw_npc_d(g);
		}else {
			win_npc_d(g);
		}
	}
	__syncthreads();
}

// execute parallel playout with all games on device mem
__global__ void repeat_playouts_npc_d(const small_board *org_b, game *games, curandState *rng,int start,int gameCount) {
	if(start >= gameCount) {
		return;
	}
	int id = get_thread_id_npc();
	// create board code
	__shared__ code_board_10 code;
	init_npc_d(&code);
	// get active game
	game *g = &games[start+id];
	__shared__ small_board sb;
	small_board *b = &sb;
		// get copy of orignal board
		copy_board_npc_d(org_b,b);
		// perform our move
		perf_move_npc_d(b,g->move,&code);
		// perform playout
		if(playout_npc_d(b,&code,&rng[id])) {
			// loss, since we moved first
			loss_npc_d(g);		
		} else {
			byte re = reason_npc_d(b);
			// could be a draw
			if(NONE == re || b->time == ELEMENT_SIZE) {
				draw_npc_d(g);
			}else {
				win_npc_d(g);
			}
		}
		__syncthreads();
}


// test kernel for 1 move
__global__ void move_npc_d(small_board *b) {
	// board code
	code_board_10 code;
	init_npc_d(&code);
	perf_move_npc_d(b,0,&code);
}

// test kernel for 1 playout
__global__ void do_playout_npc_d(small_board *org_b,long seed) {
	small_board sb;
	small_board *b = &sb;
	// get copy of orignal board
	copy_board_npc_d(org_b,b);
	// init rnd
	curandState rng;
	curand_init(seed,0,0,&rng);
	// board code
	code_board_10 code;
	init_npc_d(&code);
	playout_npc_d(b,&code,&rng);
	copy_board_npc_d(b,org_b);
}

__global__ void setup_rnd_kernel_npc(curandState* const rng,
				 const int seed)
{
	const int id = get_thread_id_npc();
	/*
     * Each thread get the same seed,
     * a different sequence number and no offset.
	 */
	curand_init(seed + id, id, 0, &rng[id]);
}

/* calculate the thread id for the current block topology */
__device__ int get_thread_id_npc() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

/**
 * CUDA Small Board Coding Impl
 */

__device__ void init_npc_d(code_board_10 *code) {
	code->lines[0].lowerbound = 0;
	code->lines[0].upperbound = 9;
	code->lines[1].lowerbound = 10;
	code->lines[1].upperbound = 20;
	code->lines[2].lowerbound = 21;
	code->lines[2].upperbound = 32;
	code->lines[3].lowerbound = 33;
	code->lines[3].upperbound = 45;
	code->lines[4].lowerbound = 46;
	code->lines[4].upperbound = 59;
	code->lines[5].lowerbound = 60;
	code->lines[5].upperbound = 74;
	code->lines[6].lowerbound = 75;
	code->lines[6].upperbound = 90;
	code->lines[7].lowerbound = 91;
	code->lines[7].upperbound = 107;
	code->lines[8].lowerbound = 108;
	code->lines[8].upperbound = 125;
	code->lines[9].lowerbound = 126;
	code->lines[9].upperbound = 144;
	code->lines[10].lowerbound = 145;
	code->lines[10].upperbound = 162;
	code->lines[11].lowerbound = 163;
	code->lines[11].upperbound = 179;
	code->lines[12].lowerbound = 180;
	code->lines[12].upperbound = 195;
	code->lines[13].lowerbound = 196;
	code->lines[13].upperbound = 210;
	code->lines[14].lowerbound = 211;
	code->lines[14].upperbound = 224;
	code->lines[15].lowerbound = 225;
	code->lines[15].upperbound = 237;
	code->lines[16].lowerbound = 238;
	code->lines[16].upperbound = 249;
	code->lines[17].lowerbound = 250;
	code->lines[17].upperbound = 260;
	code->lines[18].lowerbound = 261;
	code->lines[18].upperbound = 270;
	
	code->corners[0] = 0;
	code->corners[1] = 9;
	code->corners[2] = 126;
	code->corners[3] = 144;
	code->corners[4] = 261;
	code->corners[5] = 270;
	
	code->edges[0].edge[0] = 1;
	code->edges[0].edge[1] = 2;
	code->edges[0].edge[2] = 3;
	code->edges[0].edge[3] = 4;
	code->edges[0].edge[4] = 5;
	code->edges[0].edge[5] = 6;
	code->edges[0].edge[6] = 7;
	code->edges[0].edge[7] = 8;
	
	code->edges[1].edge[0] = 20;
	code->edges[1].edge[1] = 32;
	code->edges[1].edge[2] = 45;
	code->edges[1].edge[3] = 59;
	code->edges[1].edge[4] = 74;
	code->edges[1].edge[5] = 90;
	code->edges[1].edge[6] = 107;
	code->edges[1].edge[7] = 125;
	
	code->edges[2].edge[0] = 162;
	code->edges[2].edge[1] = 179;
	code->edges[2].edge[2] = 195;
	code->edges[2].edge[3] = 210;
	code->edges[2].edge[4] = 224;
	code->edges[2].edge[5] = 237;
	code->edges[2].edge[6] = 249;
	code->edges[2].edge[7] = 260;
	
	code->edges[3].edge[0] = 269;
	code->edges[3].edge[1] = 268;
	code->edges[3].edge[2] = 267;
	code->edges[3].edge[3] = 266;
	code->edges[3].edge[4] = 265;
	code->edges[3].edge[5] = 264;
	code->edges[3].edge[6] = 263;
	code->edges[3].edge[7] = 262;
	
	code->edges[4].edge[0] = 250;
	code->edges[4].edge[1] = 228;
	code->edges[4].edge[2] = 225;
	code->edges[4].edge[3] = 211;
	code->edges[4].edge[4] = 196;
	code->edges[4].edge[5] = 180;
	code->edges[4].edge[6] = 163;
	code->edges[4].edge[7] = 145;
	
	code->edges[5].edge[0] = 108;
	code->edges[5].edge[1] = 91;
	code->edges[5].edge[2] = 75;
	code->edges[5].edge[3] = 60;
	code->edges[5].edge[4] = 46;
	code->edges[5].edge[5] = 33;
	code->edges[5].edge[6] = 21;
	code->edges[5].edge[7] = 10;
	
	code->maxLine = 19;
	code->minLine = 0;
}


__device__ short getLine_npc_d(const code_board * code, idx index) {
	short i;
	for(i = 0; i < 19; i++) {
		if(code->lines[i].upperbound >= index && code->lines[i].lowerbound <= index) {
			return i;
		}
	}
	return -1;
}

__device__ void getNeighbours_npc_d(const code_board * code, idx index, neighbours * n) {
	int i;
	for(i=0;i<6;i++) {
		n->neighbour[i] = NO_NEIGHBOUR;
	}
	short lineNr = getLine_npc_d(code, index);
	line lineBounds;
	lineBounds= code->lines[lineNr];
	idx newIndex = index-1;
	if(newIndex >= lineBounds.lowerbound && newIndex <= lineBounds.upperbound) {
		n->neighbour[0] = newIndex;
	}
	newIndex = index+1;
	if(newIndex >= lineBounds.lowerbound && newIndex <= lineBounds.upperbound) {
		n->neighbour[1] = newIndex;
	}
	short prevLine = lineNr-1;
	if(prevLine > code->minLine) {
		line prevLineBounds = code->lines[prevLine];
		short offset2 = lineBounds.lowerbound - prevLineBounds.lowerbound;
		short offset3 = lineBounds.upperbound - prevLineBounds.upperbound;
		idx index2 = index-offset2;
		idx index3 = index-offset3;
		if(index3<index2) {
			newIndex = index2;
			index2 = index3;
			index3 = newIndex;
		}
		if(index2 >= prevLineBounds.lowerbound && index2 <= prevLineBounds.upperbound) {
			n->neighbour[2] = index2;
		}	
		if(index3 >= prevLineBounds.lowerbound && index3 <= prevLineBounds.upperbound) {
			n->neighbour[3] = index3;
		}
	}
	short nextLine = lineNr+1;
	if(nextLine < code->maxLine) {
		line nextLineBounds = code->lines[nextLine];
		short offset4 = nextLineBounds.lowerbound - lineBounds.lowerbound;
		short offset5 = nextLineBounds.upperbound - lineBounds.upperbound;
		idx index4 = index+offset4;
		idx index5 = index+offset5;
		if(index5<index4) {
			newIndex = index4;
			index4 = index5;
			index5 = newIndex;
		}
		if(index4 >= nextLineBounds.lowerbound && index4 <= nextLineBounds.upperbound) {
			n->neighbour[4] = index4;
		} 
		if(index5 >= nextLineBounds.lowerbound && index5 <= nextLineBounds.upperbound) {
			n->neighbour[5] = index5;
		} 
	}

}
/**
 * CUDA Small Board Impl
 */
__device__ byte onMove_npc_d(const small_board * b) {
	return next_npc_d(b) == WHITE ? BLACK : WHITE;
}

__device__ void setNext_npc_d(small_board * b, int color) {
	if(WHITE == color) {
		b->state = (b->state & (~128));
	} else if (BLACK == color) {
		b->state = (b->state | 128);
	}
}

__device__ byte next_npc_d(const small_board * b) {
	if((b->state & 128) == 0) {
		byte white = WHITE;
		return white;
	} else {
		byte black = BLACK;
		return black;
	}
}

__device__ void setWinner_npc_d(small_board * b, int color) {
	if(WHITE == color) {
		b->state = b->state & (~16);
	} else if (BLACK == color) {
		b->state = b->state | 16;
	}
}

__device__ byte getWinner_npc_d(const small_board * b) {
	if((b->state & 16) == 16) {
		return BLACK;
	}
	return WHITE;
}

__device__ void setReason_npc_d(small_board * b, int reason) {
	if(NONE == reason) {
		b->state = b->state & (~96);
	} else if(RING == reason) {
		b->state = b->state | 32;
	} else if(BRIDGE == reason) {
		b->state = b->state | 64;
	} else if(FORK == reason) {
		b->state = b->state | 96;
	}
}

__device__ byte reason_npc_d(const small_board * b) {
	byte reason = b->state & 96;
	if(reason == 0) {
		return NONE;
	} else if(reason == 32) {
		byte ring = RING;
		return ring;
	} else if(reason == 64) {
		byte bridge = BRIDGE;
		return bridge;
	} else if(reason == 96) {
		byte fork = FORK;
		return fork;
	} else {
		// error
	}
	return 255;
}

__device__ byte size_npc_d(const small_board * b) {
	return b->state & 15;
}

__device__ void copy_board_npc_d(const small_board *src,small_board *target) {
	target->time = src->time;
	target->last = src->last;
	target->state = src->state;

	int i;
	for(i = 0; i < ELEMENT_SIZE; i++) {
		target->cell[i] = src->cell[i];
		target->rank[i] = src->rank[i];
		target->parent[i] = src->parent[i];
	}
}

/**
 * CUDA DSF Impl
 */


// DSF make
__device__ void makeSet_npc_d(small_board * b, idx index) {
	b->parent[index] = index;
	b->rank[index] = 0;
}

/* DSF find. No recursion on CUDA
 so path compression over iteration */
__device__ idx findRoot_npc_d(small_board * b, idx index) {
	// find root
	idx newIndex = index;
	while(b->parent[newIndex] != newIndex) {
		newIndex = b->parent[newIndex];
	}
	return newIndex;
}

// DSF union
__device__ void unionSets_npc_d(small_board * b, idx index1, idx index2) {
	idx root1 = findRoot_npc_d(b, index1);
	idx root2 = findRoot_npc_d(b, index2);
	if(root1 == root2) {
		return;
	}
	if(b->rank[root1] > b->rank[root2]) {
		b->parent[root2] = root1;
	} else {
		b->parent[root1] = root2;
		if(b->rank[root1]==b->rank[root2]) {
			b->rank[root2] = b->rank[root2]+1;
		}
	}
}

/**
 * CUDA Game Impl
 */

__device__ void win_npc_d(game * g) {
	g->win = g->win+1;
}

__device__ void loss_npc_d(game * g) {
	g->loss = g->loss+1;
}

__device__ void draw_npc_d(game * g) {
	g->draw = g->draw+1;
}

/**
 * CUDA Playout Impl
 */


__device__ void perf_move_npc_d(small_board * b, idx move, const code_board * code) {
	if(b->time <= ELEMENT_SIZE) {
		// get color for this move
		byte color = onMove_npc_d(b);
		b->last = move;
		b->cell[move] = color;
		makeSet_npc_d(b,move);
		__shared__ neighbours n;
		getNeighbours_npc_d(code, move, &n);
		short i;
		// just friendly neighours and not off the board
		for(i = 0; i < 6; i++) {
			if(n.neighbour[i] != NO_NEIGHBOUR && b->cell[n.neighbour[i]] == color) {
				unionSets_npc_d(b, move, n.neighbour[i]);
			}
		}		
		b->time = b->time+1;	
		setNext_npc_d(b,color);
	} else {
		// board is full ! should not happen
	}
}

// does one playout. returns true if current player wins, otherwise false
__device__ bool playout_npc_d(small_board * b, const code_board * code,curandState * rng) {
	byte color = onMove_npc_d(b);
	__shared__ idx toPlay[ELEMENT_SIZE];
	short lastIndex = init_toplay_npc_d(toPlay,b);
	bool win = false;
	while(!win) {
		// no more free moves
		if(lastIndex < 0) {
			return false;
		}
		win = playoutStep_npc_d(b,code,toPlay, lastIndex--,rng);
		if(win) {
			byte winner = getWinner_npc_d(b);
			if(winner == color) {
				return true;
			}
			return false;
		}
	}
	return false;
}

// does one step in playout: rnd move and checking for win
// returns true if it was a winning move
__device__ bool playoutStep_npc_d(small_board * b, const code_board * code, idx* toPlay, short lastIndex,curandState * rng) {
	idx index = getIndex_npc_d(toPlay, lastIndex,rng);
	if(ringWin_npc_d(b,index,code)) {
		return true;
	}
	perf_move_npc_d(b,index,code);
	if(bridgeWin_npc_d(b,code)) {
		return true;
	}
	if(forkWin_npc_d(b,code)) {
		return true;
	}
	return false;
}

// gets a random free index
__device__ idx getIndex_npc_d(idx* toPlay, short last,curandState * rng) {
	short rnd = getRnd_npc_d(last,rng);
	idx index = toPlay[rnd];
	toPlay[rnd] = toPlay[last];
	return index;
}

__device__ short getRnd_npc_d(short last,curandState * rng) {
	if(last == 0) {
		return last;
	}
	unsigned int random_number = curand(rng);
	return random_number % last;
}

__device__ idx init_toplay_npc_d(idx* toplay, const small_board * b) {
	idx lastIndex = 0;
	short i;
	for(i = 0; i < ELEMENT_SIZE; i++) {
		if(b->cell[i] != WHITE && b->cell[i] != BLACK) {
			toplay[lastIndex] = i;
			lastIndex++;
		}
		
	}
	return lastIndex-1;
}

/**
 * CUDA Winning Conditions Impl
 */


// checks whether there is a bridge for current color
__device__ bool bridgeWin_npc_d(small_board * b, const code_board * code) {
	// move is done, so check for next = prev player
	byte color = next_npc_d(b);
	if(checkForBridge_npc_d(b,code,color)) {
		setWinner_npc_d(b, color);
		setReason_npc_d(b, BRIDGE);
		return true;
	}
	return false;
}

// checks whether there is a bridge for current color
__device__ bool checkForBridge_npc_d(small_board * b, const code_board * code, byte colorOnMove) {
	if(b->cell[code->corners[0]] == colorOnMove) {
		idx root1 = findRoot_npc_d(b, code->corners[0]);
		idx root2;
		if(b->cell[code->corners[1]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[1]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[2]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[2]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[3]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[3]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[1]] == colorOnMove) {
		idx root1 = findRoot_npc_d(b, code->corners[1]);
		idx root2;
		if(b->cell[code->corners[2]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[2]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[3]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[3]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[2]] == colorOnMove) {
		idx root1 = findRoot_npc_d(b, code->corners[2]);
		idx root2;
		if(b->cell[code->corners[3]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[3]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[3]] == colorOnMove) {
		idx root1 = findRoot_npc_d(b, code->corners[3]);
		idx root2;
		if(b->cell[code->corners[4]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[4]);
			if(root1 == root2) {
				return true;
			}
		}
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	if(b->cell[code->corners[4]] == colorOnMove) {
		idx root1 = findRoot_npc_d(b, code->corners[4]);
		idx root2;
		if(b->cell[code->corners[5]] == colorOnMove) {
			root2 = findRoot_npc_d(b, code->corners[5]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	return false;
}

// checks whether there is a fork for current color
__device__ bool forkWin_npc_d(small_board * b, const code_board * code) {
	// move is done, so check for next = prev player
	byte color = next_npc_d(b);
	if(checkForFork_npc_d(b,code,color)) {
		setWinner_npc_d(b, color);
		setReason_npc_d(b, FORK);
		return true;
	}
	return false;
}

// checks whether there is a fork for current color
__device__ bool checkForFork_npc_d(small_board * b, const code_board * code, byte colorOnMove) {
	int i;
	int j;
	// for 4 edges (if there are three edges connected it will detected within this 4 edges)
	for(i=0; i<4;i++) {
		__shared__ edge ce;
		ce = code->edges[i];
		for(j=0; j<size_npc_d(b)-2;j++) {
			// find friendly cell
			if(b->cell[ce.edge[j]] == colorOnMove) {
				// test for all edges if the cell is at least connected to 3 other edges
				if(isConnectedToThreeEdges_npc_d(ce.edge[j], i, b, colorOnMove, code)) {
					return true;
				}
			}
		}
	}
	return false;
}

__device__ bool isConnectedToThreeEdges_npc_d(idx index, short orginEdge, small_board * b, byte colorOnMove,const code_board * code) {
	byte count = 0;
	int i;
	// test all other edges besides origin edge
	for(i=0; i < 6; i++) {
		if(i != orginEdge) {
			if(isConnectedToEdge_npc_d(index,code->edges[i],b,colorOnMove)) {
				count++;
			}
		}
	}
	if(count >= 3) {
		return true;
	}
	return false;
}
	
__device__ bool isConnectedToEdge_npc_d(idx index, edge edge, small_board * b, byte colorOnMove) {
	idx root1 = findRoot_npc_d(b, index);
	int k;
	for(k = 0; k < size_npc_d(b)-2; k++) {
		if(b->cell[edge.edge[k]] == colorOnMove) {
			idx root2 = findRoot_npc_d(b, edge.edge[k]);
			if(root1 == root2) {
				return true;
			}
		}
	}
	return false;
}



__device__ bool ringWin_npc_d(small_board * b, idx mv, const code_board * code) {
	byte color = onMove_npc_d(b);
	if(checkForRing_npc_d(b,mv,code,color)) {
		setWinner_npc_d(b, color);
		setReason_npc_d(b, RING);
		// for result only
		perf_move_npc_d(b,mv,code);
		return true;
	}
	return false;
}

// ring checking requires board BEFORE setting move
__device__ bool checkForRing_npc_d(small_board * b, idx move, const code_board * code, byte color) {
	__shared__ neighbours n;
	getNeighbours_npc_d(code, move, &n);
	short i;
	short count = 0;
	for(i = 0; i < 6; i++) {
		idx new_neighbour = n.neighbour[i];
		if(new_neighbour != NO_NEIGHBOUR && b->cell[new_neighbour] == color) {
			count++;
			// test whether neighbour now has 6 friendly neighbours
			if(checkForNearRing_npc_d(b,code,new_neighbour,color)) {
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
		idx root1 = findRoot_npc_d(b, n.neighbour[0]);
		idx root2;
		if(n.neighbour[1] != NO_NEIGHBOUR) {
			if((n.neighbour[4] == NO_NEIGHBOUR || n.neighbour[5] == NO_NEIGHBOUR) && (n.neighbour[2] == NO_NEIGHBOUR || n.neighbour[3] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b, n.neighbour[1]);
				if(root1 == root2) {
					return true;
				}				
			}
		}
		if(n.neighbour[3] != NO_NEIGHBOUR) {
			if(n.neighbour[2] == NO_NEIGHBOUR && (n.neighbour[4] == NO_NEIGHBOUR||n.neighbour[5] == NO_NEIGHBOUR ||n.neighbour[1] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b,n.neighbour[3]);
				if(root1 == root2) {
					return true;
				}	
			}
		}
		if(n.neighbour[5] != NO_NEIGHBOUR) {
			if(n.neighbour[4] == NO_NEIGHBOUR&& (n.neighbour[3] == NO_NEIGHBOUR||n.neighbour[2] == NO_NEIGHBOUR ||n.neighbour[1] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b,n.neighbour[5]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	if(n.neighbour[1] != NO_NEIGHBOUR) {
		idx root1 = findRoot_npc_d(b, n.neighbour[1]);
		idx root2;
		if(n.neighbour[2] != NO_NEIGHBOUR) {
			if(n.neighbour[3] == NO_NEIGHBOUR&& (n.neighbour[4] == NO_NEIGHBOUR||n.neighbour[5] == NO_NEIGHBOUR ||n.neighbour[0] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b, n.neighbour[2]);
				if(root1 == root2) {
					return true;
				}
			}
		}
		if(n.neighbour[4] != NO_NEIGHBOUR) {
			if(n.neighbour[5] == NO_NEIGHBOUR&& (n.neighbour[0] == NO_NEIGHBOUR||n.neighbour[2] == NO_NEIGHBOUR ||n.neighbour[3] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b, n.neighbour[4]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	if(n.neighbour[2] != NO_NEIGHBOUR) {
		idx root1 = findRoot_npc_d(b, n.neighbour[2]);
		idx root2;
		if(n.neighbour[4] != NO_NEIGHBOUR) {
			if(n.neighbour[0] == NO_NEIGHBOUR&& (n.neighbour[3] == NO_NEIGHBOUR||n.neighbour[5] == NO_NEIGHBOUR ||n.neighbour[1] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b, n.neighbour[4]);
				if(root1 == root2) {
					return true;
				}
			}
		}
		if(n.neighbour[5] != NO_NEIGHBOUR) {
			if((n.neighbour[4] == NO_NEIGHBOUR || n.neighbour[0] == NO_NEIGHBOUR) && (n.neighbour[1] == NO_NEIGHBOUR || n.neighbour[3] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b, n.neighbour[5]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	if(n.neighbour[3] != NO_NEIGHBOUR) {
		idx root1 = findRoot_npc_d(b, n.neighbour[3]);
		idx root2;
		if(n.neighbour[4] != NO_NEIGHBOUR) {
			if((n.neighbour[2] == NO_NEIGHBOUR || n.neighbour[0] == NO_NEIGHBOUR) && (n.neighbour[1] == NO_NEIGHBOUR || n.neighbour[5] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b, n.neighbour[4]);
				if(root1 == root2) {
					return true;
				}
			}
		}
		if(n.neighbour[5] != NO_NEIGHBOUR) {
			if(n.neighbour[1] == NO_NEIGHBOUR&& (n.neighbour[4] == NO_NEIGHBOUR||n.neighbour[2] == NO_NEIGHBOUR ||n.neighbour[0] == NO_NEIGHBOUR)) {
				root2 = findRoot_npc_d(b, n.neighbour[5]);
				if(root1 == root2) {
					return true;
				}
			}
		}
	}
	return false;
}

__device__ bool checkForNearRing_npc_d(small_board * b,const code_board * code,idx nb, byte color) {
	__shared__ neighbours nn;
	getNeighbours_npc_d(code, nb, &nn);
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