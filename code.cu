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

#include "code.h"

void init(code_board_10 * code) {
	code->lines[0].lowerbound = 0;
	code->lines[0].upperbound = 9;
	code->lines[1].lowerbound = 10;
	code->lines[1].upperbound = 20;
	code->lines[2].lowerbound = 21;
	code->lines[2].upperbound = 32;
	code->lines[3].lowerbound = 33;
	code->lines[3].upperbound = 45;
	code->lines[4].lowerbound = 46;
	code->lines[4].upperbound = 59;
	code->lines[5].lowerbound = 60;
	code->lines[5].upperbound = 74;
	code->lines[6].lowerbound = 75;
	code->lines[6].upperbound = 90;
	code->lines[7].lowerbound = 91;
	code->lines[7].upperbound = 107;
	code->lines[8].lowerbound = 108;
	code->lines[8].upperbound = 125;
	code->lines[9].lowerbound = 126;
	code->lines[9].upperbound = 144;
	code->lines[10].lowerbound = 145;
	code->lines[10].upperbound = 162;
	code->lines[11].lowerbound = 163;
	code->lines[11].upperbound = 179;
	code->lines[12].lowerbound = 180;
	code->lines[12].upperbound = 195;
	code->lines[13].lowerbound = 196;
	code->lines[13].upperbound = 210;
	code->lines[14].lowerbound = 211;
	code->lines[14].upperbound = 224;
	code->lines[15].lowerbound = 225;
	code->lines[15].upperbound = 237;
	code->lines[16].lowerbound = 238;
	code->lines[16].upperbound = 249;
	code->lines[17].lowerbound = 250;
	code->lines[17].upperbound = 260;
	code->lines[18].lowerbound = 261;
	code->lines[18].upperbound = 270;
	
	code->corners[0] = 0;
	code->corners[1] = 9;
	code->corners[2] = 126;
	code->corners[3] = 144;
	code->corners[4] = 261;
	code->corners[5] = 270;
	
	code->edges[0].edge[0] = 1;
	code->edges[0].edge[1] = 2;
	code->edges[0].edge[2] = 3;
	code->edges[0].edge[3] = 4;
	code->edges[0].edge[4] = 5;
	code->edges[0].edge[5] = 6;
	code->edges[0].edge[6] = 7;
	code->edges[0].edge[7] = 8;
	
	code->edges[1].edge[0] = 20;
	code->edges[1].edge[1] = 32;
	code->edges[1].edge[2] = 45;
	code->edges[1].edge[3] = 59;
	code->edges[1].edge[4] = 74;
	code->edges[1].edge[5] = 90;
	code->edges[1].edge[6] = 107;
	code->edges[1].edge[7] = 125;
	
	code->edges[2].edge[0] = 162;
	code->edges[2].edge[1] = 179;
	code->edges[2].edge[2] = 195;
	code->edges[2].edge[3] = 210;
	code->edges[2].edge[4] = 224;
	code->edges[2].edge[5] = 237;
	code->edges[2].edge[6] = 249;
	code->edges[2].edge[7] = 260;
	
	code->edges[3].edge[0] = 269;
	code->edges[3].edge[1] = 268;
	code->edges[3].edge[2] = 267;
	code->edges[3].edge[3] = 266;
	code->edges[3].edge[4] = 265;
	code->edges[3].edge[5] = 264;
	code->edges[3].edge[6] = 263;
	code->edges[3].edge[7] = 262;
	
	code->edges[4].edge[0] = 250;
	code->edges[4].edge[1] = 228;
	code->edges[4].edge[2] = 225;
	code->edges[4].edge[3] = 211;
	code->edges[4].edge[4] = 196;
	code->edges[4].edge[5] = 180;
	code->edges[4].edge[6] = 163;
	code->edges[4].edge[7] = 145;
	
	code->edges[5].edge[0] = 108;
	code->edges[5].edge[1] = 91;
	code->edges[5].edge[2] = 75;
	code->edges[5].edge[3] = 60;
	code->edges[5].edge[4] = 46;
	code->edges[5].edge[5] = 33;
	code->edges[5].edge[6] = 21;
	code->edges[5].edge[7] = 10;
	
	code->maxLine = 19;
	code->minLine = 0;

}

short getLine(const code_board * code, idx index) {
	short i;
	for(i = 0; i < 19; i++) {
		if(code->lines[i].upperbound >= index && code->lines[i].lowerbound <= index) {
			return i;
		}
	}
	return ERROR;
}

void getNeighbours(const code_board * code, idx index, neighbours * n) {
	int i;
	for(i=0;i<6;i++) {
		n->neighbour[i] = NO_NEIGHBOUR;
	}

	short lineNr = getLine(code, index);
	line lineBounds = code->lines[lineNr];
	idx newIndex = index-1;
	if(newIndex >= lineBounds.lowerbound && newIndex <= lineBounds.upperbound) {
		n->neighbour[0] = newIndex;
	}
	newIndex = index+1;
	if(newIndex >= lineBounds.lowerbound && newIndex <= lineBounds.upperbound) {
		n->neighbour[1] = newIndex;
	}
	short prevLine = lineNr-1;
	if(prevLine > code->minLine) {
		line prevLineBounds = code->lines[prevLine];
		short offset2 = lineBounds.lowerbound - prevLineBounds.lowerbound;
		short offset3 = lineBounds.upperbound - prevLineBounds.upperbound;
		idx index2 = index-offset2;
		idx index3 = index-offset3;
		if(index3<index2) {
			newIndex = index2;
			index2 = index3;
			index3 = newIndex;
		}
		if(index2 >= prevLineBounds.lowerbound && index2 <= prevLineBounds.upperbound) {
			n->neighbour[2] = index2;
		}	
		if(index3 >= prevLineBounds.lowerbound && index3 <= prevLineBounds.upperbound) {
			n->neighbour[3] = index3;
		}
	}
	short nextLine = lineNr+1;
	if(nextLine < code->maxLine) {
		line nextLineBounds = code->lines[nextLine];
		short offset4 = nextLineBounds.lowerbound - lineBounds.lowerbound;
		short offset5 = nextLineBounds.upperbound - lineBounds.upperbound;
		idx index4 = index+offset4;
		idx index5 = index+offset5;
		if(index5<index4) {
			newIndex = index4;
			index4 = index5;
			index5 = newIndex;
		}
		if(index4 >= nextLineBounds.lowerbound && index4 <= nextLineBounds.upperbound) {
			n->neighbour[4] = index4;
		} 
		if(index5 >= nextLineBounds.lowerbound && index5 <= nextLineBounds.upperbound) {
			n->neighbour[5] = index5;
		} 
	}
}
