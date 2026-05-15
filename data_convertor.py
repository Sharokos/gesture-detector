import xml.etree.ElementTree as ET
from datetime import datetime
import json
import uuid

def parse_time_slots(root):
    """Build mapping from TIME_SLOT_ID -> time (in ms)."""
    time_slots = {}
    for ts in root.find("TIME_ORDER"):
        ts_id = ts.attrib["TIME_SLOT_ID"]
        ts_value = int(ts.attrib["TIME_VALUE"])
        time_slots[ts_id] = ts_value
    return time_slots


def extract_gesture_annotations(root, time_slots, tier_id="Gesture_Unit"):
    """Extract gestures from a given tier."""
    gestures = []
    
    # Find the correct tier
    for tier in root.findall("TIER"):
        if tier.attrib.get("TIER_ID") == tier_id:
            
            for i, ann in enumerate(tier.findall(".//ALIGNABLE_ANNOTATION")):
                start_id = ann.attrib["TIME_SLOT_REF1"]
                end_id = ann.attrib["TIME_SLOT_REF2"]

                start_ms = time_slots[start_id]
                end_ms = time_slots[end_id]

                gestures.append({
                    "gesture": f"gesture_{i}",
                    "start": start_ms / 1000.0,
                    "end": end_ms / 1000.0
                })

    return gestures


# JSON TO EAF

def build_time_slots(gestures):
    """Create unique TIME_SLOT entries."""
    times = set()
    for g in gestures:
        times.add(int(g["start"] * 1000))
        times.add(int(g["end"] * 1000))

    sorted_times = sorted(times)

    time_slot_map = {}
    time_order = ET.Element("TIME_ORDER")

    for i, t in enumerate(sorted_times, start=1):
        ts_id = f"ts{i}"
        time_slot_map[t] = ts_id

        ET.SubElement(time_order, "TIME_SLOT", {
            "TIME_SLOT_ID": ts_id,
            "TIME_VALUE": str(t)
        })

    return time_order, time_slot_map


def build_tier(gestures, time_slot_map, tier_id="Gesture_Unit"):
    """Build the annotation tier."""

    tier = ET.Element("TIER", {
        "LINGUISTIC_TYPE_REF": "Gesture_Unit_Timing",
        "TIER_ID": tier_id
    })

    for i, g in enumerate(gestures, start=1):
        start_ms = int(g["start"] * 1000)
        end_ms = int(g["end"] * 1000)

        ann = ET.SubElement(tier, "ANNOTATION")

        alignable = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{i}",
            "TIME_SLOT_REF1": time_slot_map[start_ms],
            "TIME_SLOT_REF2": time_slot_map[end_ms]
        })

        ET.SubElement(alignable, "ANNOTATION_VALUE").text = ""

    return tier


def json_to_eaf(gestures, output_path):
    """Main conversion function."""

    # Register xsi namespace
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "",
        "DATE": datetime.now().isoformat(),
        "FORMAT": "3.0",
        "VERSION": "3.0",
        "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation":
            "http://www.mpi.nl/tools/elan/EAFv3.0.xsd"
    })

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------

    header = ET.SubElement(root, "HEADER", {
        "MEDIA_FILE": "",
        "TIME_UNITS": "milliseconds"
    })

    urn_prop = ET.SubElement(header, "PROPERTY", {
        "NAME": "URN"
    })
    urn_prop.text = f"urn:nl-mpi-tools-elan-eaf:{uuid.uuid4()}"

    last_ann_prop = ET.SubElement(header, "PROPERTY", {
        "NAME": "lastUsedAnnotationId"
    })
    last_ann_prop.text = str(len(gestures))

    # ------------------------------------------------------------------
    # TIME_ORDER
    # ------------------------------------------------------------------

    time_order, time_slot_map = build_time_slots(gestures)
    root.append(time_order)

    # ------------------------------------------------------------------
    # TIER
    # ------------------------------------------------------------------

    tier = build_tier(gestures, time_slot_map)
    root.append(tier)

    # ------------------------------------------------------------------
    # LINGUISTIC_TYPE
    # ------------------------------------------------------------------

    linguistic_type = ET.SubElement(root, "LINGUISTIC_TYPE", {
        "GRAPHIC_REFERENCES": "false",
        "LINGUISTIC_TYPE_ID": "Gesture_Unit_Timing",
        "TIME_ALIGNABLE": "true"
    })

    # ------------------------------------------------------------------
    # WRITE FILE
    # ------------------------------------------------------------------

    tree = ET.ElementTree(root)

    tree.write(
        output_path,
        encoding="utf-8",
        xml_declaration=True
    )

def eaf_to_json(eaf_path, output_path=None):
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_slots = parse_time_slots(root)
    gestures = extract_gesture_annotations(root, time_slots)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(gestures, f, indent=2)

    return gestures
    
if __name__ == "__main__":

    with open(
        r"G:\OpenPose\detect_gestures\Emotional_LA_Pianist_L2_EM_AV_FS_P31\results\gestures.json",
        "r",
        encoding="utf-8"
    ) as f:

        gestures = json.load(f)

    json_to_eaf(gestures, "test.eaf")

    print("EAF file written to test.eaf")