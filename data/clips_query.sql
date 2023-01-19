SELECT *
FROM clips
WHERE (a_or_b_lines in ('a_lines', 'b_lines'))
    AND (frame_homogeneity IS NULL)
    AND (patient_id IS NOT NULL)
    AND (exam_id IS NOT NULL)
    AND (vid_id IS NOT NULL)
