# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:13:47 2023
akutam_fyx
"""

from hash_encoding import HashEmbedder
   
def get_embedder(args):
    embed = HashEmbedder(bounding_box=args.bounding_box, \
                         n_levels=args.hash_n_level, \
                        log2_hashmap_size=args.log2_hashmap_size, \
                        finest_resolution=args.finest_res)
    out_dim = embed.out_dim   
    return embed, out_dim