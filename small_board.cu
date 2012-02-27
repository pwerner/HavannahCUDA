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


void init_small_board(small_board * b) {
	int i;
	for(i = 0; i < ELEMENT_SIZE; i++) {
		b->cell[i] = EMPTY;
	}
	b->time = 0;
	b->last = ERROR;

	setNext(b, BLACK); 	/* white starts */
	setReason(b, NONE);
	setSize(b, 10);
}

void setSize(small_board * b, byte size) {
	if(size>15) {
		// error
	} else {
		b->state = b->state & (~15);
		b->state = b->state | size;
	}
}

byte size(const small_board * b) {
	return b->state & 15;
}

void setWinner(small_board * b, int color) {
	if(WHITE == color) {
		b->state = b->state & (~16);
	} else if (BLACK == color) {
		b->state = b->state | 16;
	}
}

byte getWinner(const small_board * b) {
	if((b->state & 16) == 16) {
		return BLACK;
	}
	return WHITE;
}

void setReason(small_board * b, int reason) {
	if(NONE == reason) {
		b->state = b->state & (~96);
	} else if(RING == reason) {
		b->state = b->state | 32;
	} else if(BRIDGE == reason) {
		b->state = b->state | 64;
	} else if(FORK == reason) {
		b->state = b->state | 96;
	}
}

byte reason(const small_board * b) {
	byte reason = b->state & 96;
	if(reason == 0) {
		return NONE;
	} else if(reason == 32) {
		byte ring = RING;
		return ring;
	} else if(reason == 64) {
		byte bridge = BRIDGE;
		return bridge;
	} else if(reason == 96) {
		byte fork = FORK;
		return fork;
	} else {
		// error
	}
	return ERROR;
}

void setNext(small_board * b, int color) {
	if(WHITE == color) {
		b->state = (b->state & (~128));
	} else if (BLACK == color) {
		b->state = (b->state | 128);
	}
}

byte next(const small_board * b) {
	if((b->state & 128) == 0) {
		byte white = WHITE;
		return white;
	} else {
		byte black = BLACK;
		return black;
	}
}

byte onMove(const small_board * b) {
	return next(b) == WHITE ? BLACK : WHITE;
}

void copy(const small_board *b,small_board *c) {
	c->time = b->time;
	c->state = b->state;
	memcpy(c->cell,b->cell,ELEMENT_SIZE);
	memcpy(c->parent,b->parent,ELEMENT_SIZE);
	memcpy(c->rank,b->rank,ELEMENT_SIZE);
}

