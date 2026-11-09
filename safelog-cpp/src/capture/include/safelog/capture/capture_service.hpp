#pragma once

#include "safelog/contracts/ports.hpp"

namespace safelog::capture {

struct NewFindingInput {
    Id inspectionId;
    Id actorId;
    std::string location;
    std::string description;
    std::string actionOpinion;
    std::string beforePhotoLocalPath;
};

class CaptureService {
public:
    CaptureService(
        IRepository& repository,
        IClock& clock,
        IIdGenerator& ids,
        IPhotoStore& photos);

    Inspection startInspection(
        const Id& siteId,
        const Id& inspectorId);

    FindingBundle addFinding(const NewFindingInput& input);

    Finding saveManualFinding(
        const Id& findingId,
        const Id& actorId,
        const std::string& description,
        const std::string& actionOpinion);

private:
    IRepository& repository_;
    IClock& clock_;
    IIdGenerator& ids_;
    IPhotoStore& photos_;
};

} // namespace safelog::capture