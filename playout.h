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

#include "game.h"
#include "winning_conditions.h"
#include <cuda.h>
#include <curand_kernel.h>

#ifndef HAVANNAH_CUDA_PLAYOUT
#define HAVANNAH_CUDA_PLAYOUT


bool playout(small_board * b, const code_board * code);
bool playoutStep(small_board * b, const code_board * code, idx* toPlay, short lastIndex);
idx getIndex(idx* toPlay, short last);
short getRnd(short last);
idx init_toplay(idx* toplay, const small_board * b);

#endif