-- FreeBird database schema v3: remove legacy blob conversation storage
--
-- All conversation data is now stored as immutable events in conversation_events.
-- The conversations table is no longer used.

DROP TABLE IF EXISTS conversations;
