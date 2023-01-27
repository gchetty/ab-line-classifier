SELECT *
FROM clips
-- a_or_b lines shouldn't be null because those clips are relevant to the experiment
WHERE (a_or_b_lines IS NOT NULL)
    -- homogeneous clips are required to correctly train the frame classifier
    AND (frame_homogeneity IS NULL)
    -- cannot correctly group frames if patient_id is null
    AND (patient_id IS NOT NULL)
    -- query_to_df function typically breaks if exam_id or vid_id is null
    AND (exam_id IS NOT NULL)
    AND (vid_id IS NOT NULL)
