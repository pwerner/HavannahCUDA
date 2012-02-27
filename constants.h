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

#ifndef HAVANNAH_CUDA_CONSTANTS
#define HAVANNAH_CUDA_CONSTANTS

typedef unsigned char byte;
typedef short idx;
// cells on board
#define ELEMENT_SIZE 271
// code for board cells
#define WHITE 0
#define BLACK 1
#define EMPTY 2
// winning reason
#define NONE 0
#define RING 1
#define BRIDGE 2
#define FORK 3
// neighbours
#define NO_NEIGHBOUR -1
// playout details
#define MAX_ACTIVE_GAMES 512
#define PLAYOUTS 20000
// error
#define ERROR 255

#endif