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

#include "small_board.h"
#include "DSF.h"
#include "code.h"

#ifndef HAVANNAH_CUDA_WINNING
#define HAVANNAH_CUDA_WINNING

/* -- Host code -- */

/* checks whether a ring could be found for move mv. Needs to be checked before move is performed */
bool ringWin(small_board * b, idx mv, const code_board * code);

/* performs the checks for ring find */
bool checkForRing(small_board * b, idx move, const code_board * code, byte color);

/* tries to find a ring that is surrounding one of the neighbours */
bool checkForNearRing(small_board * b,const code_board * code,idx nb, byte color);

/* checks whether a bridge could be found on current board for color that made the last move */
bool bridgeWin(small_board * b, const code_board * code);

/* performs the bridge check */
bool checkForBridge(small_board * b, const code_board * code, byte colorOnMove);

/* checks whether a fork could be found on current board for color that made last move */
bool forkWin(small_board * b, const code_board * code);

/* performs the check for forks */
bool checkForFork(small_board * b, const code_board * code, byte colorOnMove);

/* tries to find one path that is connected to (at least) three edges */
bool isConnectedToThreeEdges(idx index, short orginEdge, small_board * b, byte colorOnMove,const code_board * code);

/* checks whether an index is connected to an edge */
bool isConnectedToEdge(idx index, edge edge, small_board * b, byte colorOnMove);

void perf_move(small_board * b, idx move, const code_board * code);

#endif