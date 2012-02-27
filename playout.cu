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
#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>

idx getIndex(idx* toPlay, short last) {
	short rnd = getRnd(last);
	idx index = toPlay[rnd];
	toPlay[rnd] = toPlay[last];
	return index;
}

short getRnd(short last) {
	if(last == 0) {
		return last;
	}
	srand(time(0));
	int random_number = rand();
	return random_number % last;
}
// does one playout. returns true if current player wins, otherwise false
bool playout(small_board * b, const code_board * code) {
	byte color = onMove(b);
	idx toPlay[ELEMENT_SIZE];
	short lastIndex = init_toplay(toPlay,b);
	bool win = false;
	while(!win) {
		// no more free moves
		if(lastIndex < 0) {
			return false;
		}
		win = playoutStep(b,code,toPlay, lastIndex--);
		if(win) {
			byte winner = getWinner(b);
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
bool playoutStep(small_board * b, const code_board * code, idx* toPlay, short lastIndex) {
	idx index = getIndex(toPlay, lastIndex);
	if(ringWin(b,index,code)) {
		return true;
	}
	perf_move(b,index,code);
	if(bridgeWin(b,code)) {
		return true;
	}
	if(forkWin(b,code)) {
		return true;
	}
	return false;
}

short init_toplay(idx* toplay, const small_board * b) {
	short lastIndex = 0;
	idx i;
	for(i = 0; i < ELEMENT_SIZE; i++) {
		if(b->cell[i] != WHITE && b->cell[i] != BLACK) {
			toplay[lastIndex++] = i;
		}
	}
	return --lastIndex;
}