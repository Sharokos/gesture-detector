import xml.etree.ElementTree as ET
import json


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

    root = ET.Element("ANNOTATION_DOCUMENT", {
        "AUTHOR": "",
        "DATE": "",
        "FORMAT": "3.0",
        "VERSION": "3.0"
    })

    # Build structure
    time_order, time_slot_map = build_time_slots(gestures)
    root.append(time_order)

    tier = build_tier(gestures, time_slot_map)
    root.append(tier)

    # Write file
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

def eaf_to_json(eaf_path, output_path=None):
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_slots = parse_time_slots(root)
    gestures = extract_gesture_annotations(root, time_slots)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(gestures, f, indent=2)

    return gestures