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

#include "PrintBoard.h"

void print(small_board *b) { 
	char mv = onMove(b) == WHITE ? 'w' : 'b';
	char nxt = next(b) == WHITE? 'w' : 'b';
	char wn = getWinner(b) == WHITE ? 'w' : 'b';
	char rs = reason(b) == NONE ? 'N' : reason(b) == BRIDGE ? 'B' : reason(b) == FORK ? 'F' : 'R';
	printf("time %d\n", b->time);
	printf("size %d \n", size(b));
	printf("on move %c \n", mv);
	printf("next %c \n", nxt);
	printf("win %c \n",wn);
	printf("reason %c \n",rs);
	printf("last move %d\n",b->last);
}


void prettyprint(small_board *b, const code_board_10 *code) {
	int j;
	for(j = code->minLine; j < code->maxLine; j++) {
		int lower = code->lines[j].lowerbound;
		int upper = code->lines[j].upperbound;
		int rest = (19-(upper-lower));
		int i;
		for(i = 0; i < rest; i++) {
			printf(" ");
		}
		for(i = lower; i <= upper; i++) {
			char cell;
			byte cellc = b->cell[i];
			if(BLACK == cellc) {
				cell = 'b';
			} else if(WHITE == cellc) {
				cell = 'w';
			} else {
				cell = '.';
			}
			printf("%c ", cell);
		}
		for(i = 0; i < 2.8*rest;i++) {
			printf(" ");
		}
		for(i = lower; i <= upper; i++) {
			printf("%03d ", i);
		}
		printf("\n");
	}

}