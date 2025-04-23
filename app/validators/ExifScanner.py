class ExifScanner():
    def is_photo(self, Image):
        # TODO: implement and unit tests
        return True


        # try:
        #     img_info = image.info
        #     if img_info is None:
        #         logger.debug("No image info found")
        #         img_info = {}
        #     for key, value in img_info.items():
        #         print(f"IMG.Info: {key}: {value}")

        #     # check for an face and ai
        #     # TODO: exif is not working, refactor to extra function and create tests
        #     exif_data = image.getexif()
        #     if exif_data is not None:
        #         logger.debug(f"{len(exif_data)} EXIF data found")
        #         if len(exif_data) > 0:
        #             for key, val in exif_data.items():
        #                 print(f'{key}:{val}')
        #                 if key in ExifTags.TAGS:
        #                     print(f'{ExifTags.TAGS[key]}:{val}')

        #             actions_description = exif_data.get('Actions Description')
        #             if actions_description:
        #                 print(f"Actions Description: {actions_description}")
        #             else:
        #                 print("Actions Description nicht gefunden.")

        #             gps_ifd = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
        #             if gps_ifd is not None and len(gps_ifd) > 0:
        #                 logger.debug("Image probably contains GPS data, so no AI")
        #             elif "Generator" in exif_data:
        #                 msg = "Image probably AI generated"
        #                 logger.warning(msg)
        #                 token = 5
        #             elif exif_data[ExifTags.Base.Software] == "PIL" or exif_data[ExifTags.Base.HostComputer]:
        #                 msg = "Image probably generated or edited"
        #                 logger.warning(msg)
        #                 token = 5
        #             elif exif_data[ExifTags.Base.Copyright] != None:
        #                 msg = "Image is copyright protected"
        #                 logger.warning(msg)
        #                 token = 10
        #         # elif exif_data[]==:
        #         #     msg = "Image is copyright protected"
        #         #     logger.warning(msg)
        #         #     token = 10
        #     else:
        #         logger.debug("No EXIF data found")
        # except Exception as e:
        #     logger.error(f"Error while checking image EXIF data: {e}")
