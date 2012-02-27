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

#include "constants.h"

#ifndef HAVANNAH_CUDA_SMALL_BOARD
#define HAVANNAH_CUDA_SMALL_BOARD

/*board up to size 10. 1092 Bytes*/
typedef struct {
	// 271 bytes
	byte cell [ELEMENT_SIZE]; /* contains cell state (empty=0, white=1, black=2)*/
	// 542 bytes
	idx parent[ELEMENT_SIZE]; /* contains index of parent for each cell (used for dsf)*/
	// 271 bytes
	byte rank[ELEMENT_SIZE]; /* contains rank for each cell (used for dsf)*/
	
	// 2 bytes
	short time; /* number of moves already played*/
	// 2 bytes
	idx last;
	byte state; /* coded board state*/
	/* 0|00|0|0000
	   a  b c  d
	a codes next
	  0: white is next
	  1: black is next
	b codes winning reason
	  00: none
	  01: ring
	  10: bridge
	  11: fork
	c codes winner
	  0: white
	  1: black
	d codes size:
	  0-15
	*/
} small_board;

/* -- Host code -- */

/* Inits empty board */
void init_small_board(small_board * b);

/* sets the size for this board */
void setSize(small_board * b, byte size);

/* gets the size for this board */
byte size(const small_board * b);

/* sets the winner for this board (BLACK/WHITE) */
void setWinner(small_board * b, int color);

/* gets the winner for this board (BLACK/WHITE) */
byte getWinner(const small_board * b);

/* sets the winning reason for this board (NONE/RING/BRIDGE/FORK) */
void setReason(small_board * b, int reason);

/* gets the winning reason for this board (NONE/RING/BRIDGE/FORK) */
byte reason(const small_board * b);

/* sets the next color to play (BLACK/WHITE) */
void setNext(small_board * b, int color);

/* gets the next color to play (BLACK/WHITE) */
byte next(const small_board * b);

/* gets the color of the player doing the current move (BLACK/WHITE) */
byte onMove(const small_board * b);

/* copys the board */
void copy(const small_board *b,small_board *c);

#endif