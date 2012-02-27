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

#include "main.h"

int main() {
	// clock_t start, end;
	// start = clock();
	/* code for move */
	//code_board_10 code;
	//init(&code);
	small_board sb;
	small_board *b = &sb;
	/* inits the board */
	init_small_board(b);
	perf_move(b, 222, &code);
	/* test with kernel 1 */
	//rateGame1(b);
	/* test with kernel 2 */
	//rateGame2(b);
	/* test with kernel 3 */
	//rateGame3(b);
	/* test with kernel npc */
	int bestMove = rateGame_npc(b);
	//end = clock();
	//printf("Runtime %5.5f seconds.\n\n",(double)(end-start)/CLOCKS_PER_SEC);
	/* wait for input */
	getchar();
	return 0;
}


