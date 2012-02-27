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

#include "DSF.h"

// DSF make
void makeSet(small_board * b, idx index) {
	b->parent[index] = index;
	b->rank[index] = 0;
}

/* DSF find. No recursion on CUDA.
So path compression over iteration */
idx findRoot(small_board * b, idx index) {
	// find root
	idx newIndex = index;
	while(b->parent[newIndex] != newIndex) {
		newIndex = b->parent[newIndex];
	}
	idx root = newIndex;
	// set root
	newIndex = index;
	while(b->parent[newIndex] != newIndex) {
		idx oldIndex = newIndex;
		newIndex = b->parent[oldIndex];
		b->parent[oldIndex] = root;
	}
	return root;
	/* 2nd variante: extra array with idx for pointer alternative (needs 542 bytes more mem!) */
}

// DSF union
void unionSets(small_board * b, idx index1, idx index2) {
	idx root1 = findRoot(b, index1);
	idx root2 = findRoot(b, index2);
	if(root1 == root2) {
		return;
	}
	if(b->rank[root1] > b->rank[root2]) {
		b->parent[root2] = root1;
	} else {
		b->parent[root1] = root2;
		if(b->rank[root1]==b->rank[root2]) {
			b->rank[root2] = b->rank[root2]+1;
		}
	}
}