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

#include "playout.h"

// test kernel for 1 move
__global__ void move_npc_d(small_board * b);
// test kernel for 1 playout
__global__ void do_playout_npc_d(small_board * b, long seed);

/* real kernel */
__global__ void playouts_npc_d(const small_board * b, game * g, curandState * rng);
__global__ void repeat_playouts_npc_d(const small_board * org_b, game * games, curandState * rng, int start, int gameCount);

/* kernel for initialize rng */
__global__ void setup_rnd_kernel_npc(curandState* rng, int seed);

/* helper for thread id */
__device__ int get_thread_id_npc();

/**
 * Small Board Coding Device Code
 */

/* codes every line with index bounds, adds the corners and codes the edges */
__device__ void init_npc_d(code_board_10 * code);

/* gets the line for a cell index */
__device__ short getLine_npc_d(const code_board * code, idx index);

/* gets the neighbours for a cell index */
__device__ void getNeighbours_npc_d(const code_board * code, idx index, neighbours * n);

/**
 * DSF Device code
 */

/* init DSF with one node on index */
__device__ void makeSet_npc_d(small_board * b, idx index);

/* finds the root of node with given index */
__device__ idx findRoot_npc_d(small_board * b, idx index);

/* unions the two sets represented by index1 and index2 */
__device__ void unionSets_npc_d(small_board * b, idx index1, idx index2);

/**
 * Game Device code
 */

__device__ void win_npc_d(game * g);
__device__ void loss_npc_d(game * g);
__device__ void draw_npc_d(game * g);

/**
 * Playout Device code
 */

__device__ short init_toplay_npc_d(idx* toplay, const small_board * b);
__device__ short getRnd_npc_d(short last,curandState * rng);
__device__ idx getIndex_npc_d(idx* toPlay, short last,curandState * rng);
__device__ bool playoutStep_npc_d(small_board * b, const code_board * code, idx* toPlay, short lastIndex,curandState * rng);
__device__ bool playout_npc_d(small_board * b, const code_board * code,curandState * rng);
__device__ void perf_move_npc_d(small_board * b, idx move, const code_board * code);

/**
 * Small Board Device code
 */



/* gets the color of the player doing the current move (BLACK/WHITE) */
__device__ byte onMove_npc_d(const small_board * b);

/* sets the next color to play (BLACK/WHITE) */
__device__ void setNext_npc_d(small_board * b, int color);

/* gets the next color to play (BLACK/WHITE) */
__device__ byte next_npc_d(const small_board * b);

/* sets the winner for this board (BLACK/WHITE) */
__device__ void setWinner_npc_d(small_board * b, int color);

/* gets the winner for this board (BLACK/WHITE) */
__device__ byte getWinner_npc_d(const small_board * b);

/* sets the winning reason for this board (NONE/RING/BRIDGE/FORK) */
__device__ void setReason_npc_d(small_board * b, int reason);

/* gets the winning reason for this board (NONE/RING/BRIDGE/FORK) */
__device__ byte reason_npc_d(const small_board * b);

/* gets the size for this board */
__device__ byte size_npc_d(const small_board * b);

/* copys the board */
__device__ void copy_board_npc_d(const small_board *src, small_board *target);

/**
 * Winning Conditions Device code
 */


/* checks whether a ring could be found for move mv. Needs to be checked before move is performed */
__device__ bool ringWin_npc_d(small_board * b, idx mv, const code_board * code);

/* performs the checks for ring find */
__device__ bool checkForRing_npc_d(small_board * b, idx move, const code_board * code, byte color);

/* tries to find a ring that is surrounding one of the neighbours */
__device__ bool checkForNearRing_npc_d(small_board * b,const code_board * code,idx neighbour, byte color);

/* checks whether a bridge could be found on current board for color that made the last move */
__device__ bool bridgeWin_npc_d(small_board * b, const code_board * code);

/* performs the bridge check */
__device__ bool checkForBridge_npc_d(small_board * b, const code_board * code, byte colorOnMove);

/* checks whether a fork could be found on current board for color that made last move */
__device__ bool forkWin_npc_d(small_board * b, const code_board * code);

/* performs the check for forks */
__device__ bool checkForFork_npc_d(small_board * b, const code_board * code, byte colorOnMove);

/* tries to find one path that is connected to (at least) three edges */
__device__ bool isConnectedToThreeEdges_npc_d(idx index, short orginEdge, small_board * b, byte colorOnMove,const code_board * code);

/* checks whether an index is connected to an edge */
__device__ bool isConnectedToEdge_npc_d(idx index, edge edge, small_board * b, byte colorOnMove);