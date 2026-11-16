"""List the region names exposed by the selected Julich-Brain statistical map."""
import argparse
import siibra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default="mni152")
    parser.add_argument("--parcellation", default="julich")
    args = parser.parse_args()

    with siibra.QUIET:
        maps = siibra.get_map(space=args.space, parcellation=args.parcellation, maptype="statistical")
    print(f"Map: {maps}")
    print(f"Number of volumes: {len(maps)}")
    for i in range(len(maps)):
        region = maps.get_region(volume=i)
        if region is not None:
            print(f"{i:03d}\t{getattr(region, 'name', str(region))}\t{region}")


if __name__ == "__main__":
    main()
