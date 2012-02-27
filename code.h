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

#ifndef HAVANNAH_CUDA_CODE
#define HAVANNAH_CUDA_CODE

/* Codes a line. Size: 4 byte */
typedef struct {
	short lowerbound;
	short upperbound;
} line;

/* Codes an edge. Size: 16 bytes */
typedef struct {
	idx edge[8];
} edge;

/* Contains coding for elements on board. Size: 188 bytes */
typedef struct {
	line lines [19];
	idx corners[6];
	edge edges[6];
	short maxLine;
	short minLine;
} code_board_10;

typedef code_board_10 code_board;

/* Neigbour of a cell. Size: 12 bytes */
typedef struct {
	idx neighbour[6];
} neighbours;

/* -- Host code -- */

/* codes every line with index bounds, adds the corners and codes the edges */
void init(code_board_10 * code);

/* gets the line for a cell index */
short getLine(const code_board * code, idx index);

/* gets the neighbours for a cell index */
void getNeighbours(const code_board * code, idx index, neighbours * n);

#endif