# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict

import cv2
from typing import Any

import numpy as np

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class RegionCounter(BaseSolution):
    """
    A class for real-time counting of objects within user-defined regions in a video stream.

    This class inherits from `BaseSolution` and provides functionality to define polygonal regions in a video frame,
    track objects, and count those objects that pass through each defined region. Useful for applications requiring
    counting in specified areas, such as monitoring zones or segmented sections.

    Attributes:
        region_template (dict): Template for creating new counting regions with default attributes including name,
            polygon coordinates, and display colors.
        counting_regions (list): List storing all defined regions, where each entry is based on `region_template`
            and includes specific region settings like name, coordinates, and color.
        region_counts (dict): Dictionary storing the count of objects for each named region.

    Methods:
        add_region: Add a new counting region with specified attributes.
        process: Process video frames to count objects in each region.
        initialize_regions: Initialize zones to count the objects in each one. Zones could be multiple as well.

    # Examples:
    #     Initialize a RegionCounter and add a counting region
    #     >>> counter = RegionCounter()
    #     >>> counter.add_region("Zone1", [(100, 100), (200, 100), (200, 200), (100, 200)], (255, 0, 0), (255, 255, 255))
    #     >>> results = counter.process(frame)
    #     >>> print(f"Total tracks: {results.total_tracks}")
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the RegionCounter for real-time object counting in user-defined regions."""
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "counts": 0,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []
        self.initialize_regions()
        self.counted_ids = []
        self.classwise_count = defaultdict(lambda: {"IN":0 })  # Dictionary for counts, categorized by class
        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.total_count =defaultdict(lambda: {"LEFT LANE": 0, "RIGHT LANE": 0})  # Dictionary for counts, categorized by class
        self.margin = self.line_width * 2


    def add_region(
        self,
        name: str,
        polygon_points: list[tuple],
        region_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> dict[str, Any]:
        """
        Add a new region to the counting list based on the provided template with specific attributes.

        Args:
            name (str): Name assigned to the new region.
            polygon_points (list[tuple]): List of (x, y) coordinates defining the region's polygon.
            region_color (tuple[int, int, int]): BGR color for region visualization.
            text_color (tuple[int, int, int]): BGR color for the text within the region.

        Returns:
            (dict[str, any]): Returns a dictionary including the region information i.e. name, region_color etc.
        """
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)
        return region

    def initialize_regions(self):
        """Initialize regions only once."""
        if self.region is None:
            self.initialize_region()
        if not isinstance(self.region, dict):  # Ensure self.region is initialized and structured as a dictionary
            self.region = {"Region#01": self.region}
        for i, (name, pts) in enumerate(self.region.items()):
            region = self.add_region(name, pts, colors(i, True), (255, 255, 255))
            region["prepared_polygon"] = self.prep(region["polygon"])

    def count_objects(
            self,
            current_centroid,
            track_id,
            prev_position,
            cls,
    ) -> None:
        """
        Count objects within a polygonal or linear region based on their tracks.

        Args:
            current_centroid (tuple[float, float]): Current centroid coordinates (x, y) in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (tuple[float, float], optional): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.

        # Examples:
        #     >>> counter = ObjectCounter()
        #     >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
        #     >>> box = [130, 230, 150, 250]
        #     >>> track_id_num = 1
        #     >>> previous_position = (120, 220)
        #     >>> class_to_count = 0  # In COCO model, class 0 = person
        #     >>> counter.count_objects((140, 240), track_id_num, previous_position, class_to_count)
        """
        if prev_position is None or track_id in self.counted_ids:
            return

        for region in self.counting_regions:
            if region["prepared_polygon"].contains(self.Point(current_centroid)):
                # Determine motion direction for vertical or horizontal polygons
                region_width = max(p[0] for p in region["polygon"].exterior.coords) - min(p[0] for p in region["polygon"].exterior.coords)
                region_height = max(p[1] for p in region["polygon"].exterior.coords) - min(p[1] for p in region["polygon"].exterior.coords)

                if (
                        region_width < region_height
                        and current_centroid[0] > prev_position[0]
                        or region_width >= region_height
                        and current_centroid[1] > prev_position[1]
                ):  # Moving right or downward

                    self.classwise_count[region["name"]]["IN"] += 1
                    self.total_count[self.names[cls]]["LEFT LANE"] += 1

                else:  # Moving left or upward

                    self.classwise_count[region["name"]]["IN"] += 1
                    self.total_count[self.names[cls]]["RIGHT LANE"] += 1

                self.counted_ids.append(track_id)

    def display_counts(self, plot_im) -> None:
        """
        Display object counts on the input image or frame.

        Args:
            plot_im (np.ndarray): The image or frame to display counts on.

        # Examples:
        #     >>> counter = ObjectCounter()
        #     >>> frame = cv2.imread("image.jpg")
        #     >>> counter.display_counts(frame)
        # """
        labels_dict = {
            "TOTAL" : f"{'RIGHT LANE: ' + str(value['RIGHT LANE']) if self.show_in else ''} "
            f"{'LEFT LANE: ' + str(value['LEFT LANE']) if self.show_out else ''}".strip()
            for key, value in self.total_count.items()
            if value["RIGHT LANE"] != 0 or value["LEFT LANE"] != 0 and (self.show_in or self.show_out)
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)




    def process(self, im0: np.ndarray) -> SolutionResults:
        """
        Process the input frame to detect and count objects within each defined region.

        Args:
            im0 (np.ndarray): Input image frame where objects and regions are annotated.

        Returns:
            (SolutionResults): Contains processed image `plot_im`, 'total_tracks' (int, total number of tracked objects),
                and 'region_counts' (dict, counts of objects per region).
        """
        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.region_initialized = True

        for box, cls, track_id, conf in zip(self.boxes, self.clss, self.track_ids, self.confs):
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            # center = self.Point(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            center = self.Point(((box[0] + box[2]) / 2, (box[3])))

            self.store_tracking_history(track_id, box)
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)


            for region in self.counting_regions:
                if region["prepared_polygon"].contains(center):
                    region["counts"] += 1
                    self.region_counts[region["name"]] = region["counts"]

        # Display region counts
        for region in self.counting_regions:
            poly = region["polygon"]
            pts = list(map(tuple, np.array(poly.exterior.coords, dtype=np.int32)))
            (x1, y1), (x2, y2) = [(int(poly.centroid.x), int(poly.centroid.y))] * 2
            self.annotator.draw_region(pts, region["region_color"], self.line_width * 2)
            self.annotator.adaptive_label(
                [x1, y1, x2, y2],
                label=str(self.classwise_count[region["name"]]["IN"]),
                color=region["region_color"],
                txt_color=region["text_color"],
                margin=self.line_width * 4,
                shape="rect",
            )
            region["counts"] = 0  # Reset for next frame
        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)

        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), region_counts=self.region_counts, result = self.classwise_count)

if __name__ == '__main__':
    cap = cv2.VideoCapture("../SAM/org_video.mp4")
    region_points = {
        "region-01": [[607, 347], [690, 351], [290, 639], [52, 632]],
        "region-02":[[687, 352], [763, 361], [516, 639], [294, 637]],
        "region-03": [[769, 361], [859, 371], [751, 652], [521, 642]],
        "region-04":[[993, 372], [1084, 374], [1290, 642], [1086, 644]],
        "region-05": [[1086, 372], [1161, 372], [1479, 644], [1295, 641]],
        "region-06":[[1164, 369], [1252, 374], [1683, 644], [1490, 642]],
    }

    # Video writer
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("../SAM/region_counting.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize region counter object
    regioncounter = RegionCounter(
        show=True,  # display the frame
        region=region_points,  # pass region points
        model = r"runs/detect/train/weights/best.pt"

    )




    counter = 0
    # Process video
    while cap.isOpened():
        success, im0 = cap.read()

        if counter < 100:
            counter +=1
            continue
        if counter > 400:
            break


        if not success:
            print("Video frame is empty or processing is complete.")
            break

        results = regioncounter(im0)
        video_writer.write(results.plot_im)
        counter += 1
        print(dict(results.result))


    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()  # destroy all opened windows