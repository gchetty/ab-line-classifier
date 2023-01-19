SELECT *
FROM clips
WHERE (a_or_b_lines IS NOT NULL)
    AND (frame_homogeneity IS NULL)
    AND (patient_id IS NOT NULL)
    AND (exam_id IS NOT NULL)
    AND (vid_id IS NOT NULL)
LIMIT 100
