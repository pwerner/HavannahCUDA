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

#ifndef HAVANNAH_CUDA_DSF
#define HAVANNAH_CUDA_DSF

/* -- Host code -- */

/* init DSF with one node on index */
void makeSet(small_board * b, idx index);

/* finds the root of node with given index */
idx findRoot(small_board * b, idx index);

/* unions the two sets represented by index1 and index2 */
void unionSets(small_board * b, idx index1, idx index2);

#endif